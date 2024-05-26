import { Elysia, t } from "elysia";
import { cors } from "@elysiajs/cors";
import { JsonOutputParser } from "@langchain/core/output_parsers";
import { END, StateGraph } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from "langchain/prompts";
import { HumanMessage } from "langchain/schema";

interface KeywordAnalysis {
  main_keyword: string;
  keywords: string;
  lsi_keywords: string;
  lt_keywords: string;
}

interface OutlinePoint {
  outline_content: string;
}

interface Outline {
  outline: OutlinePoint[];
}

const app = new Elysia()
  .post(
    "/generate-article",
    async ({ body }) => {
      console.log(body.query);
      const agentState = {
        query: "",
        searchResult: "",
        keywordAnalysis: "",
        articleOutline: "",
        articleGeneratorStep: {
          value: (x: number) => x + 1,
          default: () => 0,
        },
        article: {
          value: (content: string, contentToAdd: string) =>
            content.concat(contentToAdd),
          default: () => "",
        },
      };

      const llm = new ChatOpenAI({
        model: "gpt-4o",
        temperature: 0,
      });

      const graph = new StateGraph({
        channels: agentState,
      });

      const webResearcherAgent = async (state: any) => {
        const res = await fetch("https://api.tavily.com/search", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            api_key: process.env.TAVILY_API_KEY,
            query: state.query,
            max_results: 3,
            include_raw_content: true,
            exclude_domains: [
              "https://en.wikipedia.org",
              "https://it.wikipedia.org",
            ],
          }),
        });
        const response: any = await res.json();
        return {
          searchResult: response.results,
        };
      };

      const keywordAnalyzerFormatInstructions =
        "Respond with a valid JSON object, containing four fields: 'main_keyword', 'keywords', 'lsi_keywords' and 'lt_keywords'.";

      const keywordAnalyzerParser = new JsonOutputParser<KeywordAnalysis>();

      const keywordAnalyzerPrompt = ChatPromptTemplate.fromMessages([
        [
          "system",
          `You excel at analyzing text. You must provide the answer in ITALIAN. Use the researcher's information to analyze the provided articles. Focus on: 'main keyword', 'LSI (Latent Semantic Indexing) keywords' and 'long tail keywords'. \n{format_instructions}`,
        ],
        new MessagesPlaceholder("messages"),
      ]);

      const keywordAnalyzerPartialPrompt = await keywordAnalyzerPrompt.partial({
        format_instructions: keywordAnalyzerFormatInstructions,
      });

      const keywordAnalyzerChain = async (state: any) => {
        const res = await keywordAnalyzerPartialPrompt
          .pipe(llm)
          .pipe(keywordAnalyzerParser)
          .invoke({
            messages: new HumanMessage(JSON.stringify(state.searchResult)),
          });
        return {
          keywordAnalysis: res,
        };
      };

      const outlineGeneratorFormatInstructions =
        "Respond with a valid JSON object, containing one field array: 'outline'.";

      const outlineGeneratorParser = new JsonOutputParser<Outline>();

      const outlineGeneratorPrompt = ChatPromptTemplate.fromMessages([
        [
          "system",
          `You excel at generating perfect blog article outline. You must provide the answer in ITALIAN. Use the researcher's information and the keyword analysis provided by the text analyzer to create the best possible struture for an article outline. It should not contain more than 6 points. \n{format_instructions}`,
        ],
        new MessagesPlaceholder("messages"),
      ]);

      const outlineGeneratorPartialPrompt =
        await outlineGeneratorPrompt.partial({
          format_instructions: outlineGeneratorFormatInstructions,
        });

      const outlineGeneratorChain = async (state: any) => {
        const res = await outlineGeneratorPartialPrompt
          .pipe(llm)
          .pipe(outlineGeneratorParser)
          .invoke({
            messages: new HumanMessage(
              `query: ${state.query}. \nsearchResult: ${JSON.stringify(
                state.searchResult
              )}\nkeywordAnalysis: ${JSON.stringify(state.keywordAnalysis)}`
            ),
          });
        return {
          articleOutline: res,
        };
      };

      const blogGeneratorCycle = async (state: any) => {
        if (
          state.articleOutline.outline.length - 1 >=
          state.articleGeneratorStep
        ) {
          return "continue";
        }
        return "end";
      };

      const paragraphGeneratorPrompt = ChatPromptTemplate.fromMessages([
        [
          "system",
          `You excel at writing perfect SEO and readable paragraph blog post. You must provide the answer in ITALIAN. You must write at least 350 words. Use the researcher's information, the keyword analysis provided by the text analyzer and write the best paragraph you can with the provided paragraph title. Include the title in your answer. Answer in Markdown format.`,
        ],
        new MessagesPlaceholder("messages"),
      ]);

      const paragraphGeneratorNode = async (state: any) => {
        const res = await paragraphGeneratorPrompt.pipe(llm).invoke({
          messages: new HumanMessage(
            `paragraph title: ${JSON.stringify(
              state.articleOutline.outline[state.articleGeneratorStep]
            )} \nsearchResult: ${JSON.stringify(
              state.searchResult
            )}\nkeywordAnalysis: ${JSON.stringify(state.keywordAnalysis)}`
          ),
        });
        return {
          articleGeneratorStep: state.articleGeneratorStep,
          article: res.content,
        };
      };

      graph.addNode("web_researcher", webResearcherAgent);
      graph.addNode("keyword_analyzer", keywordAnalyzerChain);
      graph.addNode("outline_generator", outlineGeneratorChain);
      graph.addNode("paragraph_generator", paragraphGeneratorNode);
      graph.addEdge("web_researcher", "keyword_analyzer");
      graph.addEdge("keyword_analyzer", "outline_generator");
      graph.addEdge("outline_generator", "paragraph_generator");
      graph.addConditionalEdges("paragraph_generator", blogGeneratorCycle, {
        continue: "paragraph_generator",
        end: END,
      });

      graph.setEntryPoint("web_researcher");

      const app = graph.compile();

      const stream = await app.stream(
        { query: body.query },
        { recursionLimit: 100 }
      );
      const encoder = new TextEncoder();
      const transfromStream = new TransformStream({
        transform(chunk, controller) {
          const text = JSON.stringify(chunk);
          controller.enqueue(encoder.encode(text));
        },
      });
      return new Response(stream.pipeThrough(transfromStream), {
        status: 200,
        headers: {
          "Content-Type": "application/octet-stream",
          "Transfer-Encoding": "chunked",
        },
      });
    },
    {
      body: t.Object({
        query: t.String(),
      }),
      type: "application/json",
    }
  )
  .use(cors());

app.listen(3000);
