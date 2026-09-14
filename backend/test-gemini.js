require("dotenv").config();

const { GoogleGenAI } = require("@google/genai");

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
});

async function test() {
  try {
    const response = await ai.models.generateContent({
      model: "gemini-3.6-flash",
      contents: "Say hello in one sentence.",
    });

    console.log("API WORKING");
    console.log(response.text);
  } catch (error) {
    console.error("API FAILED");
    console.error(error.message);
  }
}

test();