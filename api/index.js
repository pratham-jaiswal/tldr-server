import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ClerkExpressRequireAuth } from "@clerk/clerk-sdk-node";
import { clerkMiddleware } from "@clerk/express";
import { ChatOpenAI } from "@langchain/openai";
import { ChatCohere } from "@langchain/cohere";
import { rateLimit } from "express-rate-limit";
import RedisStore from "rate-limit-redis";
import * as cheerio from "cheerio";
import validator from "validator";
import "@dotenvx/dotenvx/config";
import mongoose from "mongoose";
import { Webhook } from "svix";
import express from "express";
import helmet from "helmet";
import Redis from "ioredis";
import axios from "axios";
import cors from "cors";

const redisClient = new Redis(process.env.UPSTASH_URL);

const app = express();
app.use(express.json());

app.use(
  clerkMiddleware({
    authorizedParties: [process.env.ALLOWED_ORIGIN],
    secretKey: process.env.CLERK_SECRET_KEY,
  })
);
app.use(helmet());

app.get("/health", (req, res) => {
  return res.json({ status: "ok" });
});

const corsOptions = {
  origin: `${process.env.ALLOWED_ORIGIN}`,
  credentials: true,
};

app.use(cors(corsOptions));

function verifyWebhook(req, res, next) {
  const payload = req.body;
  const headers = req.headers;
  const wh = new Webhook(process.env.SVIX_SECRET);

  try {
    let msg = wh.verify(JSON.stringify(payload), headers);
  } catch (err) {
    return res.status(400).json({});
  }

  next();
}

const userRateLimiter = rateLimit({
  store: new RedisStore({
    sendCommand: (...args) => redisClient.call(...args),
  }),
  windowMs: 15 * 60 * 1000,
  max: 20,
  keyGenerator: (req) => {
    const cookieStr = req.headers.cookie;
    if (cookieStr) {
      const cookies = Object.fromEntries(
        cookieStr.split(";").map((cookie) => {
          const [key, value] = cookie.trim().split("=");
          return [key, value];
        })
      );

      const token = cookies["__session"];
      if (token) {
        try {
          const base64Url = sessionJWT.split(".")[1];
          const base64 = base64Url.replace(/-/g, "+").replace(/_/g, "/");
          const jsonPayload = decodeURIComponent(
            atob(base64)
              .split("")
              .map((c) => `%${`00${c.charCodeAt(0).toString(16)}`.slice(-2)}`)
              .join("")
          );
          const payload = JSON.parse(jsonPayload);
          const userId = payload.userId;
          if (userId) return userId;
        } catch (e) {
          console.error("JWT decode error:", e);
        }
      }
    }

    return req.ip;
  },
  handler: (req, res) => {
    res
      .status(429)
      .json({ error: "Too many requests. Please try again later." });
  },
});

app.use("/validate-user", verifyWebhook);
app.use("/tldr/url", [ClerkExpressRequireAuth(), userRateLimiter]);
app.use("/tldr/text", [ClerkExpressRequireAuth(), userRateLimiter]);

mongoose
  .connect(`${process.env.MONGODB_URI}`, { family: 4 })
  .then(() => {
    console.log("Connected to MongoDB");
  })
  .catch((err) => console.error("Error connecting to MongoDB:", err));

const tldrSchema = new mongoose.Schema({
  url: {
    type: String,
    required: true,
    unique: true,
  },
  tldr: {
    type: String,
    required: true,
  },
});
const TLDR = mongoose.model("TLDR", tldrSchema);

const userSchema = new mongoose.Schema({
  fullName: {
    type: String,
    required: true,
  },
  email: {
    type: String,
    required: true,
    unique: true,
  },
  userId: {
    type: String,
    required: true,
    unique: true,
  },
  createdAt: {
    type: Date,
  },
});

const User = mongoose.model("User", userSchema);

function fetchWebpageContent(url) {
  return axios
    .get(`https://api.allorigins.win/get?url=${url}`)
    .then((response) => {
      const $ = cheerio.load(response.data.contents);

      $("script, style, noscript").remove();

      const text = $("body").text().replace(/\s+/g, " ").trim();

      return text;
    })
    .catch((error) => {
      console.error("Error fetching the webpage:", error);
      return null;
    });
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function generateTLDR(content, provider) {
  const llm =
    provider === "OpenAI GPT-4o mini"
      ? new ChatOpenAI({
          model: "gpt-4o-mini",
          temperature: 0,
        })
      : new ChatCohere({
          model: "command-r",
          temperature: 0,
          maxRetries: 1,
        });

  return llm
    .invoke([
      {
        role: "system",
        content: `You're a TL;DR AI.
                  Your task is to provide a clear, concise summary prioritizing relevant concepts. 
                  - Summarize the core points clearly.
                  - Exclude section headers unless necessary for clarity.
                  - Do not list all sections; instead, synthesize the content into essential insights.
                  - Avoid repetition and unnecessary details.
                  - Omit irrelevant details.
                  - Do not answer questions.
                  - Do not add your own opinion, views, or censoring.
                  - If the input is a question, summarize it as "User asked a question about [topic]."
                  - Respond in bullet points without without filler phrases.`,
      },
      { role: "user", content: content },
    ])
    .then((response) => {
      return response.content;
    })
    .catch((error) => {
      console.error("Error generating TL;DR:", error);
      return null;
    });
}

async function summarizeWebpage(url, provider, content) {
  try {
    if (!content && !url) throw new Error("No content to summarize");

    if (url) {
      if (url) content = await fetchWebpageContent(url);
      if (!content) throw new Error("Couldn't fetch webpage content");
    }

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 2048,
      chunkOverlap: 200,
    });

    const chunks = await textSplitter.splitText(content);

    let chunkTLDRs = await Promise.all(
      chunks.map(async (chunk) => {
        await delay(1000);
        return generateTLDR(chunk, provider);
      })
    );

    if (chunkTLDRs.length === 0) throw new Error("No chunked TL;DRs found");
    const finalTLDR = await generateTLDR(chunkTLDRs.join(" "), provider);
    if (!finalTLDR) throw new Error("Error generating final TL;DR");

    return finalTLDR;
  } catch (error) {
    console.error("Error summarizing webpage:", error);
    return null;
  }
}

app.post("/validate-user", async (req, res) => {
  const { data, object, type } = req.body;
  const {
    id,
    first_name,
    last_name,
    email_addresses,
    primary_email_address_id,
    created_at,
  } = data;

  if (object === "event" && type === "user.created") {
    const primary_email = email_addresses.find(
      (email) => primary_email_address_id === email.id
    );
    const email = primary_email ? primary_email.email_address : null;

    const existingUser = await User.findOne({ userId: id });
    if (!existingUser) {
      const newUser = new User({
        fullName: `${first_name} ${last_name}`,
        email: email,
        userId: id,
        createdAt: created_at,
      });
      await newUser.save();
    }
  } else if (object === "event" && type === "session.created") {
    const existingUser = await User.findOne({ userId: id });
    if (!existingUser) {
      return res.status(401).json({ error: "Unauthorized" });
    }
  }

  return res.json({ status: "success" });
});

app.post("/tldr/url", async (req, res) => {
  const { url, provider, useKnowledgeHub, shouldSave } = req.body;

  if (
    !url ||
    !provider ||
    typeof useKnowledgeHub === "undefined" ||
    typeof shouldSave === "undefined"
  )
    return res.status(400).json({ error: "Insufficient parameters." });

  if (!validator.isURL(url, { require_protocol: true })) {
    return res.status(400).json({ error: "Invalid or missing URL." });
  }

  try {
    let existingTLDR = await TLDR.findOne({ url });
    if (useKnowledgeHub && existingTLDR) {
      return res.json({ tldr: existingTLDR.tldr });
    }

    limiter(req, res, async (err) => {
      if (err) return next(err);

      const tldr = await summarizeWebpage(url, provider, null);
      if (!tldr)
        return res.status(500).json({ error: "Error generating a TL;DR" });

      if (shouldSave) {
        existingTLDR = await TLDR.findOneAndUpdate(
          { url },
          { tldr },
          { upsert: true, new: true }
        );
      }

      return res.status(200).json({ tldr: tldr });
    });
  } catch (error) {
    return res
      .status(400)
      .json({ error: "URL does not exist or is unreachable." });
  }
});

app.post("/tldr/text", async (req, res) => {
  const { content, provider } = req.body;

  if (!content || !provider)
    return res.status(400).json({ error: "Insufficient parameters." });

  try {
    const tldr = await summarizeWebpage(null, provider, content);
    if (!tldr) {
      return res.status(500).json({ error: "Error generating a TL;DR" });
    }

    return res.status(200).json({ tldr: tldr });
  } catch (error) {
    console.error("Error processing TL;DR request:", error);
    return res.status(500).json({ error: "Internal server error." });
  }
});

app.get("/fetch-saved-tldrs", async (req, res) => {
  try {
    const savedTLDRs = await TLDR.find().sort({ _id: -1 });
    return res.status(200).json({ savedTLDRs });
  } catch (error) {
    console.error("Error fetching saved TL;DRs:", error);
    return res.status(500).json({ error: "Internal server error." });
  }
});

const port = 3001;
app.listen(port, () => {
  console.log(`Server running`);
});
