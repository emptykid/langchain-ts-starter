import * as dotenv from "dotenv";
// import { OpenAI } from "langchain";

import {CSVLoader} from "langchain/document_loaders";
import {HNSWLib} from "langchain/vectorstores";
import {OpenAIEmbeddings} from "langchain/embeddings";

dotenv.config();

const loader = new CSVLoader("files/test.csv", "内容");
const docs = await loader.load();
// console.log(docs);

const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());

const result = await vectorStore.similaritySearch("风骚", 1);
console.log(result);

/*
const model = new OpenAI({
  modelName: "gpt-3.5-turbo",
  openAIApiKey: process.env.OPENAI_API_KEY,
});

const res = await model.call(
  "What's a good idea for an application to build with GPT-3?"
);

console.log(res);
 */