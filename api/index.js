import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ClerkExpressRequireAuth } from "@clerk/clerk-sdk-node";
import { clerkMiddleware } from "@clerk/express";
import { ChatOpenAI } from "@langchain/openai";
import { ChatCohere } from "@langchain/cohere";
import { rateLimit } from "express-rate-limit";
import RedisStore from "rate-limit-redis";
import querystring from "querystring";
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

const app = express();
app.use(express.json());

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
  patreonDetails: {
    patreonId: {
      type: String,
      required: true,
    },
    tier: {
      type: String,
      required: true,
    },
  },
  createdAt: {
    type: Date,
  },
});

const User = mongoose.model("User", userSchema);

app.get("/", (req, res) => {
  res.send("Hi");
});

app.get("/favicon.ico", (req, res) => {
  res.status(204).end();
});

app.get("/favicon.png", (req, res) => {
  res.status(204).end();
});

const redisClient = new Redis(process.env.UPSTASH_URL, {
  tls: {},
  reconnectOnError: () => true,
});

redisClient.on("error", (err) => {
  console.error("Redis error:", err);
});

app.use(
  clerkMiddleware({
    authorizedParties: [
      process.env.ALLOWED_ORIGIN,
      new URL(process.env.PATREON_REDIRECT_URI).origin,
    ],
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

function extractUserIdFromToken(req, res, next) {
  try {
    if (!req.headers) {
      return next();
    }

    const cookieStr = req.headers.cookie;
    if (!cookieStr) {
      req.headers["tldr-ai-clerk-user-id"] = null;
      return next();
    }
    const cookies = Object.fromEntries(
      cookieStr.split(";").map((cookie) => {
        const [key, value] = cookie.trim().split("=");
        return [key, value];
      })
    );
    const token = cookies?.["__session"];
    if (!token) {
      req.headers["tldr-ai-clerk-user-id"] = null;
      return next();
    }

    const base64Url = token.split(".")[1];
    const base64 = base64Url.replace(/-/g, "+").replace(/_/g, "/");
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split("")
        .map((c) => `%${`00${c.charCodeAt(0).toString(16)}`.slice(-2)}`)
        .join("")
    );
    const payload = JSON.parse(jsonPayload);
    const userId = payload.userId;
    req.headers["tldr-ai-clerk-user-id"] = userId;

    return next();
  } catch (error) {
    req.headers["tldr-ai-clerk-user-id"] = null;
    return next();
  }
}

app.use(extractUserIdFromToken);

const rateLimits = {
  free: 1,
  supporter: 3,
  "digital insider": 10,
};

async function applyDynamicRateLimit(req, res, next) {
  if (!req.headers) {
    return next();
  }

  const userId = req.headers?.["tldr-ai-clerk-user-id"];
  if (!userId) {
    req.headers["tldr-ai-clerk-max-rate-limit"] = "1";
    return next();
  }

  try {
    const user = await User.findOne({ userId });

    const tier = user?.patreonDetails?.tier?.toLowerCase();
    const maxRequests = rateLimits[tier] || 1;

    req.headers["tldr-ai-clerk-max-rate-limit"] = maxRequests.toString();
  } catch (err) {
    console.error("Rate limit tier lookup failed:", err);
    req.headers["tldr-ai-clerk-max-rate-limit"] = "1";
  }

  next();
}

app.use(applyDynamicRateLimit);

const userRateLimiter = rateLimit({
  store: new RedisStore({
    sendCommand: (...args) => redisClient.call(...args),
  }),
  windowMs: 6 * 60 * 60 * 1000,
  max: (req) => {
    const parsed = parseInt(req.headers["rate-limit-max"] || "1", 10);
    return isNaN(parsed) ? 1 : parsed;
  },
  keyGenerator: (req) => {
    if (req.headers?.["tldr-ai-clerk-user-id"]) {
      return req.headers["tldr-ai-clerk-user-id"];
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
app.use("/tldr/url", ClerkExpressRequireAuth());
app.use("/tldr/text", [ClerkExpressRequireAuth(), userRateLimiter]);

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

function generateTLDR(content, model) {
  const llm = new ChatOpenAI({
    model: model,
    temperature: 0,
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

async function summarizeWebpage(url, model, content) {
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
        return generateTLDR(chunk, model);
      })
    );

    if (chunkTLDRs.length === 0) throw new Error("No chunked TL;DRs found");
    const finalTLDR = await generateTLDR(chunkTLDRs.join(" "), model);
    if (!finalTLDR) throw new Error("Error generating final TL;DR");

    return finalTLDR;
  } catch (error) {
    console.error("Error summarizing webpage:", error);
    return null;
  }
}

const validModels = ["gpt-4o-mini", "gpt-4.1-nano"];

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
        patreonDetails: {},
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
  const { url, model, useKnowledgeHub, shouldSave } = req.body;

  if (
    !url ||
    !model ||
    typeof useKnowledgeHub === "undefined" ||
    typeof shouldSave === "undefined"
  )
    return res.status(400).json({ error: "Insufficient parameters." });

  if (!validModels.includes(model)) {
    return res.status(400).json({ error: "Invalid model." });
  }

  if (!validator.isURL(url, { require_protocol: true })) {
    return res.status(400).json({ error: "Invalid or missing URL." });
  }

  try {
    let existingTLDR = await TLDR.findOne({ url });
    if (useKnowledgeHub && existingTLDR) {
      return res.json({ tldr: existingTLDR.tldr });
    }

    userRateLimiter(req, res, async (err) => {
      if (err) return next(err);

      const tldr = await summarizeWebpage(url, model, null);
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
  const { content, model } = req.body;

  if (!content || !model)
    return res.status(400).json({ error: "Insufficient parameters." });

  if (!validModels.includes(model)) {
    return res.status(400).json({ error: "Invalid model." });
  }

  try {
    const tldr = await summarizeWebpage(null, model, content);
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

app.get("/auth/patreon", (req, res) => {
  const url = `https://www.patreon.com/oauth2/authorize?response_type=code&client_id=${
    process.env.PATREON_CLIENT_ID
  }&redirect_uri=${encodeURIComponent(
    process.env.PATREON_REDIRECT_URI
  )}&scope=identity%20identity.memberships`;

  return res.redirect(url);
});

app.get("/connect/patreon/callback", async (req, res) => {
  const code = req.query.code;

  try {
    const tokenRes = await axios.post(
      "https://www.patreon.com/api/oauth2/token",
      querystring.stringify({
        code,
        grant_type: "authorization_code",
        client_id: process.env.PATREON_CLIENT_ID,
        client_secret: process.env.PATREON_CLIENT_SECRET,
        redirect_uri: process.env.PATREON_REDIRECT_URI,
      }),
      { headers: { "Content-Type": "application/x-www-form-urlencoded" } }
    );

    const accessToken = tokenRes.data.access_token;

    const identityRes = await axios.get(
      "https://www.patreon.com/api/oauth2/v2/identity?include=memberships&fields[user]=email,first_name,full_name,image_url,last_name,social_connections,thumb_url,url,vanity",
      {
        headers: { Authorization: `Bearer ${accessToken}` },
      }
    );

    const member_id = identityRes?.data?.included?.[0]?.id;

    if (member_id) {
      const memberUrl = `https://www.patreon.com/api/oauth2/v2/members/${member_id}?include=currently_entitled_tiers&fields%5Btier%5D=title`;
      const response = await axios.get(memberUrl, {
        headers: {
          Authorization: `Bearer ${process.env.PATREON_CREATOR_ACCESS_TOKEN}`,
        },
      });

      const tier = response?.data?.included?.[0]?.attributes?.title;
      await User.findOneAndUpdate(
        { userId: req.headers["tldr-ai-clerk-user-id"] },
        { $set: { patreonDetails: { patreonId: member_id, tier } } },
        { upsert: true, new: true }
      );
    }

    return res.redirect(`${process.env.ALLOWED_ORIGIN}`);
  } catch (err) {
    console.error("Patreon OAuth error:", err.response?.data || err.message);
    return res.status(500).send("OAuth failed");
  }
});

app.get("/get-user", async (req, res) => {
  try {
    const existingUser = await User.findOne({
      userId: req.headers["tldr-ai-clerk-user-id"],
    });

    if (!existingUser) {
      return res.status(404).json({ error: "User not found" });
    }

    const tier = existingUser.patreonDetails?.tier?.toLowerCase();
    if (!tier || !rateLimits[tier]) {
      return res.status(400).json({ error: "Invalid or missing tier" });
    }

    const userDetails = {
      ...existingUser.toObject(),
      patreonDetails: {
        ...existingUser.patreonDetails,
        rateLimit: rateLimits[tier],
      },
    };

    return res.json({ userDetails: userDetails });
  } catch (error) {
    console.error("Error fetching user:", error);
    return res.status(500).json({ error: "Internal Server Error" });
  }
});

app.get("/diconnect-patreon", async (req, res) => {
  try {
    const result = await User.findOneAndUpdate(
      { userId: req.headers["tldr-ai-clerk-user-id"] },
      { $set: { patreonDetails: {} } },
      { upsert: true, new: true }
    );

    return res.json({ userDetails: result });
  } catch (error) {
    console.error("Error fetching user:", error);
    return res.status(500).json({ error: "Internal Server Error" });
  }
});

const port = 3001;
app.listen(port, () => {
  console.log(`Server running`);
});
