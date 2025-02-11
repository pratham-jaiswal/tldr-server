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

const app = express();
app.use(express.json());

const corsOptions = {
  origin: `${process.env.ALLOWED_ORIGIN}`,
};
app.use(cors(corsOptions));

app.use((req, res, next) => {
  const apiKey = req.get("x-api-key");
  const referrer = req.get("Referer") || "";
  const userAgent = req.get("User-Agent") || "";
  const validApiKey = process.env.API_KEY;

  const allowedReferrer = process.env.ALLOWED_ORIGIN;

  const blockedAgents = [/Mozilla/, /Chrome/, /Safari/, /Firefox/, /Edge/];

  if (
    apiKey !== validApiKey ||
    // !referrer.startsWith(allowedReferrer) ||
    blockedAgents.some((pattern) => pattern.test(userAgent))
  ) {
    return res.status(403).json({ error: "403 Forbidden" });
  }

  next();
});

const limiter = rateLimit({
  windowMs: 1 * 60 * 60 * 1000,
  max: 2,
  message: "You're only allowed to make two api request per hour.",
  statusCode: 429,
  standardHeaders: "draft-8",
  keyGenerator: (req) => req.ip,
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
    provider === "openai"
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
        content: `Provide a clear, concise summary prioritizing relevant concepts. 
                    Omit irrelevant details. Respond in bullet points without filler phrases.`,
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

    let chunkTLDRs = [];

    await chunks.reduce(async (promiseChain, chunk) => {
      await promiseChain;
      await delay(1000);
      const generatedTLDR = await generateTLDR(chunk, provider);
      chunkTLDRs.push(generatedTLDR);
    }, Promise.resolve());

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
});

app.post("/tldr/text", async (req, res) => {
  const { content, provider } = req.body;

  if (!content || !provider)
    return res.status(400).json({ error: "Insufficient parameters." });

  const tldr = await summarizeWebpage(null, provider, content);
  if (!tldr) return res.status(500).json({ error: "Error generating a TL;DR" });

  res.json({ tldr: tldr });
});

const port = 3000;
app.listen(port, () => {
  console.log(`Server running`);
});
