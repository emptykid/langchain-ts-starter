import * as dotenv from "dotenv";
// import { OpenAI } from "langchain";

import {CSVLoader} from "langchain/document_loaders/fs/csv";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import {OpenAIEmbeddings} from "langchain/embeddings/openai";
import {RecursiveCharacterTextSplitter} from "langchain/text_splitter";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import {OpenAI} from "langchain/llms/openai";
import { SelfQueryRetriever } from "langchain/retrievers/self_query";
import { FunctionalTranslator } from "langchain/retrievers/self_query/functional";
import { AttributeInfo } from "langchain/schema/query_constructor";
import { RetrievalQAChain } from "langchain/chains";
import { ContextualCompressionRetriever } from "langchain/retrievers/contextual_compression";
import { LLMChainExtractor } from "langchain/retrievers/document_compressors/chain_extract";
import { ChatOpenAI } from "langchain/chat_models/openai";
import {AIChatMessage, HumanChatMessage, SystemChatMessage} from "langchain/schema";
import { parse } from 'csv-parse';
import * as fs from "fs";
import * as readline from "readline";


dotenv.config();

type Message = {
    message: string;
    time: string;
}

type MessageMap = {
    [uid: string]: string[];
}


/**
 * 加载csv文件，转换为消息格式
 * @param file
 * @returns {Promise<MessageMap>}
 */
async function loadMessage(file: string): Promise<MessageMap> {
    return new Promise((resolve) => {
        const messageMap: MessageMap = {};
        const readInterface = readline.createInterface({
            input: fs.createReadStream(file),
        });
        readInterface.on('line', (line) => {
            // 切分每一列
            const columns = line.split(',');
            const uid = columns[1];
            const message = columns[5];
            if (!messageMap[uid]) {
                messageMap[uid] = [];
            }
            messageMap[uid].unshift(message);
        });
        readInterface.on('close', () => {
            resolve(messageMap);
        });
    });
}

async function main(file: string) {
    console.log(`加载文件：${file}`);
    const message = await loadMessage(file);
    // console.log(message);
    let maxLength = 0;
    const keys = Object.keys(message);
    const msgLengthMap = {
        5: 0,
        10: 0,
        20: 0,
        50: 0,
        100: 0,
        300: 0,
    };
    let cameraCount = 0;
    let allCameraCount = 0;
    const largeMessageArr: string[][] = [];
    Object.values(message).forEach((value) => {
        maxLength = Math.max(maxLength, value.length);
        if (value.length > 100) {
            largeMessageArr.push(value);
        }
        // 数量分布
        if (value.length < 5) {
            msgLengthMap[5] += 1;
        } else if (value.length < 10) {
            msgLengthMap[10] += 1;
        } else if (value.length < 20) {
            msgLengthMap[20] += 1;
        } else if (value.length < 50) {
            msgLengthMap[50] += 1;
        } else if (value.length < 100) {
            msgLengthMap[100] += 1;
        } else {
            msgLengthMap[300] += 1;
        }
        // 包含拍题提问
        if (value.some((item) => item.includes("https"))) {
            cameraCount += 1;
        }
        // 全部都是拍题的数量
        if (value.every((item) => item.includes("https"))) {
            allCameraCount += 1;
        }
    });
    console.log(`总用户数: ${keys.length}, 单用户最大提问条数: ${maxLength}`);
    console.log(`包含拍题的用户数量：${cameraCount}， 占比：${ratio(cameraCount, keys.length)}`);
    console.log(`全部都是拍题的用户数量：${allCameraCount}， 占比：${ratio(allCameraCount, keys.length)}`);
    console.log("消息数量分布：");
    // eslint-disable-next-line guard-for-in
    for (const key in msgLengthMap) {
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        console.log(`<${key}:\t${msgLengthMap[key]}\t${ratio(msgLengthMap[key], keys.length)}`);
    }
    /*
    console.log("单用户大于100条消息的内容：");
    largeMessageArr.forEach((value) => {
        console.log("=====================================");
        console.log(value.join("\n"));
    });
     */
}

function ratio(num: number, total: number): string {
    return `${(num * 100 / total).toFixed(2)}%`;
}



async function run() {
    await main("files/msg_atp_output_scene_20230617.csv");
    console.log("\n\n");
    await main("files/msg_atp_output_scene_20230618.csv");
}

//run();

const loader = new CSVLoader("files/test.csv", "内容");
const docs = await loader.load();
console.log(docs);
const items: string[] = [];

docs.forEach((doc, index) => {
    if (index > 40 && index < 80) {
        items.push(doc.pageContent);
    }
});
// console.log(items);

const systemPrompt = `
你是一个内容分析助手，帮我对用户的提问进行归类。类别有如下这些：
数学问题，语文问题，拍照提问，翻译，其它。内容含有『英文』的都归到翻译类别，带数字计算的都归到数学问题。 
用JSON格式输出，比如：
输入： 
四季和天气的英文单词
礼物的英文
2/9+4/5+7/9+1/5=用简便计算
https://zhiji-bot-1253445850.cos.ap-beijing.myqcloud.com/resources/ec0660d63afbc62ff8accecfc37f7faf.jpg
任二向量、，若||=||=1，且cos∠(，)=，则= . ( )A. 对B. 错
用英文写一句话文案
学习的成语
https://zhiji-bot-1253445850.cos.ap-beijing.myqcloud.com/resources/faeed97aa8e7dbd78426bf19c0fa02ae.jpg
给小红书写一个有关日常标题 14个字  积极感性
普高的高考有30分的优惠政策，我应该选择哪里

输出：
{
    "数学问题": [
        "任二向量、，若||=||=1，且cos∠(，)=，则= . ( )A. 对B. 错",
        "2/9+4/5+7/9+1/5=用简便计算",
    ]
    "语文问题": [
        "学习的成语",
    ],
    "翻译": [
        "四季和天气的英文单词",
        "礼物的英文",
        "用英文写一句话文案",
    ],
    "拍照提问": [
        "https://zhiji-bot-1253445850.cos.ap-beijing.myqcloud.com/resources/ec0660d63afbc62ff8accecfc37f7faf.jpg",
"https://zhiji-bot-1253445850.cos.ap-beijing.myqcloud.com/resources/faeed97aa8e7dbd78426bf19c0fa02ae.jpg",
    ],
    "其它": [
        "给小红书写一个有关日常标题 14个字  积极感性",
        "普高的高考有30分的优惠政策，我应该选择哪里",
    ]
}

现在开始归类下面内容：
`;



const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo-16k"
});

const ret: AIChatMessage = await model.call([
    new HumanChatMessage(systemPrompt + items.join("\n"))
]);
console.log(ret.text);

/*
// console.log(docs);

const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
});

const splitDocs = await textSplitter.splitDocuments(docs);

const vectorStore = await HNSWLib.fromDocuments(splitDocs, new OpenAIEmbeddings());
await vectorStore.save('./indexes');

const result = await vectorStore.similaritySearch("风骚", 1);
console.log(result);
 */

/*
const vectorStore = await HNSWLib.load('./indexes', new OpenAIEmbeddings());
const model = new OpenAI();
const baseCompressor = LLMChainExtractor.fromLLM(model);
const retriever = new ContextualCompressionRetriever({
    baseCompressor,
    baseRetriever: vectorStore.asRetriever(),
});
const chain = RetrievalQAChain.fromLLM(model, retriever);

const res = await chain.call({
    query: "帮我对这些内容做一下归类",
});

console.log({ res });
 */

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