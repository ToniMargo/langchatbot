// chatbot.js (final, merged prompt for Gemini with deserialization fix)

import dotenv from "dotenv";
dotenv.config();

//console.log("GOOGLE_API_KEY:", process.env.GOOGLE_API_KEY || "NOT FOUND");

import { v4 as uuidv4 } from "uuid";
import readline from "readline";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { START, END, StateGraph, MemorySaver } from "@langchain/langgraph";
import {
  HumanMessage,
  AIMessage,
  SystemMessage,
} from "@langchain/core/messages";
import { z } from "zod";

// Initialize Gemini model
const llm = new ChatGoogleGenerativeAI({
  model: "gemini-2.0-flash",
  temperature: 0.3,
  apiKey: process.env.GOOGLE_API_KEY,
});

// Gemini expects a single string, so we'll merge the chat history
const callModel = async (state) => {
  const allMessages = (state.messages ?? [])
    .filter(Boolean)
    .map((msg) => {
      if (msg.type === "human") return new HumanMessage(msg.content);
      if (msg.type === "ai") return new AIMessage(msg.content);
      if (msg.type === "system") return new SystemMessage(msg.content);
      return null;
    })
    .filter(Boolean);

  if (!allMessages || allMessages.length === 0) {
    console.error("callModel received state with no messages:", state);
    throw new Error("No messages found. User input may be missing.");
  }

  //console.log("DEBUG - deserialized state.messages:", allMessages);

  const mergedPrompt = allMessages
    .map((msg) => {
      const type = msg._getType();
      if (type === "human") return `User: ${msg.content}`;
      if (type === "ai") return `Assistant: ${msg.content}`;
      if (type === "system") return `System: ${msg.content}`;
      return "";
    })
    .filter(Boolean)
    .join("\n")
    .trim();

  //console.log("Merged prompt to Gemini:\n" + mergedPrompt);

  if (!mergedPrompt) {
    throw new Error("Prompt is empty. Cannot invoke Gemini with no content.");
  }

  const response = await llm.invoke(mergedPrompt);
  console.log("Gemini response:", response.content);
  return {
    messages: [...allMessages, new AIMessage(response.content)],
    language: state.language,
  };
};

// Define schema with Zod
const ChatbotStateSchema = z.object({
  messages: z.array(
    z.object({
      type: z.enum(["human", "ai", "system"]),
      content: z.string(),
    })
  ),
  language: z.string(),
});

const workflow = new StateGraph(ChatbotStateSchema)
  .addNode("model", callModel)
  .addEdge(START, "model")
  .addEdge("model", END);

const app = workflow.compile({ checkpointer: new MemorySaver() });

// CLI setup
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
const thread_id = uuidv4();
const config = { configurable: { thread_id } };

let conversationState = {
  messages: [new SystemMessage("You are a helpful assistant.")],
  language: "English",
};

const promptUser = () => {
  rl.question("You: ", async (input) => {
    if (!input.trim()) {
      console.log("Please enter a message.");
      return promptUser();
    }

    const newMessages = [
      ...conversationState.messages,
      new HumanMessage(input),
    ];
    const serializedMessages = newMessages.map((msg) => ({
      type: msg._getType(),
      content: msg.content,
    }));

    const nextState = {
      messages: serializedMessages,
      language: conversationState.language,
    };

    //console.log("Invoking app with nextState.messages:", nextState.messages);

    const result = await app.invoke(nextState, config);

    const reply =
      result.messages && result.messages.length > 0
        ? result.messages[result.messages.length - 1]
        : {
            type: "ai",
            content: "I'm sorry, I didn't get a response. Try again?",
          };

    //console.log("Assistant:", reply.content);

    const deserialized = result.messages
      .map((msg) => {
        if (msg.type === "human") return new HumanMessage(msg.content);
        if (msg.type === "ai") return new AIMessage(msg.content);
        if (msg.type === "system") return new SystemMessage(msg.content);
        return null;
      })
      .filter(Boolean);

    conversationState = {
      messages: deserialized,
      language: nextState.language,
    };

    promptUser();
  });
};

console.log("Chatbot started. Type your message below:");
promptUser();
// This code initializes a chatbot using the Google Gemini model, allowing users to interact with it via the command line.
