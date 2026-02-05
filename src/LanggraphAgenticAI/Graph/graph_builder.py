from langgraph.graph import StateGraph, START, END
from src.LanggraphAgenticAI.state.state import State
from src.LanggraphAgenticAI.nodes.basic_chatbot_node import BasicChatbotNode
from src.LanggraphAgenticAI.tools.search_tool import get_tools, create_tool_node
from langgraph.prebuilt import tools_condition,ToolNode
from src.LanggraphAgenticAI.nodes.chatbot_with_tool_node import ChatbotWithToolNode

class GraphBuilder:
    def __init__(self, model):
        self.llm=model
        self.graph_builder=StateGraph(State)

    def basic_chatbot_build_graph(self):
        """
        Builds a basic chatbot graph using LangGraph.
        This method initializes a chatbot node using the 'BasicChatbotNode' class
        and integrates it into the graph. The chatbot node is set as both the 
        entry and exit point of the graph.
        """ 

        self.basic_chatbot_node=BasicChatbotNode(self.llm)

        self.graph_builder.add_node('chatbot', self.basic_chatbot_node.process)
        self.graph_builder.add_edge(START, "chatbot")
        self.graph_builder.add_edge("chatbot", END)


    def chatbot_with_tool_build_graph(self):
        """
        Builds a chatbot with tool graph using LangGraph.
        This method initializes a chatbot node with tool-using capabilities
        and integrates it into the graph. The chatbot node is set as both the 
        entry and exit point of the graph.
        """ 
        ## Define the tool and tool node
        tools=get_tools()
        tool_node=create_tool_node(tools)
        ## Define the LLM
        llm=self.llm
        ## Define the chatbot node
        chatbot_with_tool_node=ChatbotWithToolNode(llm)
        chatbot_node=chatbot_with_tool_node.create_chatbot(tools)
        
        ## Add nodes
        self.graph_builder.add_node("chatbot",chatbot_node)
        self.graph_builder.add_node("tools",tool_node)
        ## Define conditional and direct edges
        self.graph_builder.add_edge(START, "chatbot")
        self.graph_builder.add_conditional_edges("chatbot",tools_condition)
        self.graph_builder.add_edge("tools","chatbot")
        self.graph_builder.add_edge("chatbot", END)

    def ai_news_builder_graph(self):

        ## added the nodes
        self.graph_builder.add_node("fetch_news","")
        self.graph_builder.add_node("summarize_news","")
        self.graph_builder.add_node("save_result", "")

        ## added the edges
        self.graph_builder.set_entry_point("fetch_news")
        self.graph_builder.add_edge("fetch_news","summarize_news")
        self.graph_builder.add_edge("summarize_news", "save_result")
        self.graph_builder.add_edge("save_result", END)

    def setup_graph(self, usecase: str):
        """
        Sets up the graph for the selected use case.
        """
        if usecase == "Basic Chatbot":
            self.basic_chatbot_build_graph()
        if usecase == "Chatbot with Tool":
            self.chatbot_with_tool_build_graph()

        return self.graph_builder.compile()
            