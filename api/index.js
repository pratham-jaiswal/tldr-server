import express from "express";
import cors from "cors";
import axios from "axios";
import * as cheerio from "cheerio";
import { ChatOpenAI } from "@langchain/openai";
import { ChatCohere } from "@langchain/cohere";
import "@dotenvx/dotenvx/config";
import { rateLimit } from "express-rate-limit";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import mongoose from "mongoose";
import cookieParser from "cookie-parser";
import mongoSanitize from "express-mongo-sanitize";
import helmet from "helmet";
import validator from "validator";

const app = express();
app.use(express.json());
app.use(cookieParser());
app.use(helmet());
app.use(mongoSanitize());

app.get("/health", (req, res) => {
  res.json({ status: "ok" });
});

const corsOptions = {
  origin: `${process.env.ALLOWED_ORIGIN}`,
  credentials: true,
};

app.use(cors(corsOptions));

app.use((req, res, next) => {
  const apiKey = req.get("x-api-key");
  const referrer = req.get("Referer") || "";
  const validApiKey = process.env.API_KEY;

  const allowedReferrer = process.env.ALLOWED_ORIGIN;
  if (apiKey !== validApiKey || !referrer.startsWith(allowedReferrer)) {
    return res.status(403).json({ error: "403 Forbidden" });
  }

  next();
});

app.use((req, res, next) => {
  if (!req.cookies.user_id) {
    const userId = Math.random().toString(36).substring(2);
    res.cookie("user_id", userId, {
      httpOnly: true,
      secure: true,
      sameSite: "None",
      maxAge: 6 * 60 * 60 * 1000,
    });
    req.userId = userId;
  } else {
    req.userId = req.cookies.user_id;
  }
  next();
});

const limiter = rateLimit({
  windowMs: 6 * 60 * 60 * 1000,
  max: 2,
  message: "You're only allowed to make two api request per hour.",
  statusCode: 429,
  standardHeaders: "draft-8",
  keyGenerator: (req) => (req.cookies.user_id ? req.cookies.user_id : req.ip),
});

app.use("/tldr/text", limiter);

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

function fetchWebpageContent(url) {
  return axios
    .get(url)
    .then((response) => {
      const $ = cheerio.load(response.data);

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
    await axios.head(url, { timeout: 5000 });

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

      res.json({ tldr: tldr });
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

    res.json({ tldr: tldr });
  } catch (error) {
    console.error("Error processing TL;DR request:", error);
    return res.status(500).json({ error: "Internal server error." });
  }
});

app.get("/fetch-saved-tldrs", async (req, res) => {
  const savedTLDRs = await TLDR.find();
  res.json({ savedTLDRs });
});

const port = 3001;
app.listen(port, () => {
  console.log(`Server running`);
});
