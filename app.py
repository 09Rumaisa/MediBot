# Imports
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import base64
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
import os
from openai import OpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
import streamlit as st
import tempfile
import os
from PIL import Image





load_dotenv()
#loadkeys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
tavil_api_key = os.getenv("TAVIL_API_KEY")

if not OPENAI_API_KEY or not langsmith_api_key or not tavil_api_key:  
    raise ValueError(" set in the environment variables.")
else:
    print("API keys loaded successfully.")

os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
project_name = "MediBot"
os.environ["LANGCHAIN_PROJECT"] = project_name
print("✅ LangSmith tracing enabled under project:", project_name)



from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import tool


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)



@tool
def fallback_tool(input: str) -> str:
    """Fallback LLM response when all else fails."""
    fallback_llm = ChatOpenAI(model="gpt-4.1-nano")  # Use cheaper or different model
    return fallback_llm.invoke(f"Fallback response for: {input}")





@tool
def analyze_medical_image(image_path: str) -> str:
    """
    Takes the path of a medical image (e.g., prescription or diagnostic report), extracts text,
    and explains the diagnosis and medications in simple terms.
    """

    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode()

        # Step 1: Extract text from the image
        vision_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract the medical text from this image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ]
        )

        extracted_text = vision_response.choices[0].message.content

        # Step 2: Analyze and explain the diagnosis and medication
        explanation_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a kind and helpful medical assistant. Explain diagnosis and medication in simple language for a non-doctor."},
                {"role": "user", "content": f"This was extracted from the image:\n\n{extracted_text}\n\nCan you explain the diagnosis and medications simply?"}
            ]
        )

        return explanation_response.choices[0].message.content

    except Exception as e:
        return f"⚠️ Failed to process the image. Error: {str(e)}"



@tool
def check_symptoms_conversational(symptoms: str) -> str:
    """Check symptoms and continue conversation to clarify before concluding."""
    loader = CSVLoader(file_path="C:\\Users\\rumai\\OneDrive\\Desktop\\MediBot\\dataset.csv")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  
    openai_api_key=OPENAI_API_KEY,
)
    vectorstore = Chroma(persist_directory="symptom_index", embedding_function=embeddings)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-4.1-nano"),
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return qa_chain.run({"question": symptoms})




tavily_tool = TavilySearchResults(k=3,tavily_api_key=os.getenv("TAVIL_API_KEY"))  
@tool
def get_drug_info(drug: str) -> str:
    """Gives information related to a drug from drugs.com. Falls back to fallback_tool if not found."""
    url = f"https://www.drugs.com/{drug.lower()}.html"

    try:
        loader = WebBaseLoader(url)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        all_splits = text_splitter.split_documents(data)

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY,
        )
        vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=embeddings,
            persist_directory="./drugdb"
        )

        llm = ChatOpenAI(model="gpt-4.1-nano")
        retriever = vectorstore.as_retriever()

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": ChatPromptTemplate.from_template("""
You are a helpful medical assistant. Use the information below to answer the question.

Context: {context}

Question: {question}

Helpful Answer:""")
            }
        )

        result = qa_chain.invoke(drug)["result"]

        if not result.strip() or "No information found" in result:
            raise ValueError("Insufficient info")

        return result

    except Exception as e:
        print(f"⚠ Drug info retrieval failed. Using fallback_tool. Error: {str(e)}")
        return fallback_tool.run(f"Provide information about the drug: {drug}")






@tool
def get_first_aid_info(emergency: str) -> str:
    """
    Fetches offline-friendly summaries and protocols for first aid emergencies like burns, choking, or seizures.
    """
    query = f"first aid protocol for {emergency} site:mayoclinic.org OR site:nhs.uk OR site:redcross.org"
    results = tavily_tool.run(query)
    return "\n\n".join([r["content"] for r in results])








# Define a general triage prompt
triage_prompt = ChatPromptTemplate.from_template(
    """You are a helpful and careful virtual medical assistant. A user will describe their symptoms in free text.

Your task:
1. Gently assess the symptoms.
2. Provide general medical advice based on commonly known guidelines (do not diagnose).
3. Warn the user if the symptoms might indicate a serious condition.
4. Encourage the user to visit a doctor if anything sounds unusual, worsening, or concerning.

Patient input:
"{input}"

Now respond with a short and clear message in under 4 sentences."""
)
# Connect the chain
llm = ChatOpenAI(model="gpt-4.1-nano")  # Replace with any other model if needed
triage_chain = triage_prompt | llm | StrOutputParser()

# Example input
user_input = "I've been coughing for three days and have a runny nose."

# Invoke the agent
response = triage_chain.invoke({"input": user_input})
print(response)
@tool
def general(input: str) -> str:
    """Give general medical advice based on user-described symptoms (no diagnosis)."""
    return triage_chain.invoke({"input": input})







llm = ChatOpenAI(model="gpt-4.1-nano")  # Replace with any other model if needed
agent = initialize_agent(
    tools=[general, check_symptoms_conversational,analyze_medical_image,get_drug_info,get_first_aid_info,fallback_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    #memory=memory,
    verbose=True,
)



client = OpenAI(api_key=OPENAI_API_KEY)  

def run_medibot(user_query: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",  # or "gpt-4.1-nano"
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": user_query}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print("⚠ Primary OpenAI client failed, using fallback...")
        return fallback_tool.run(user_query)







def gui():


    st.set_page_config(page_title="🧠 MediBot - Chatbot", layout="centered")
    st.markdown("## 🩺 MediBot - Your AI Medical Assistant")
    st.markdown("Upload a medical image (e.g., prescription) and ask questions about it.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "image_context" not in st.session_state:
        st.session_state.image_context = None

    # Upload image file
    uploaded_image = st.file_uploader("📎 Upload a medical image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Medical Image", use_column_width=True)
        with st.spinner("🔍 Extracting text and context from the image..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    tmp_file.write(uploaded_image.read())
                    tmp_path = tmp_file.name

                # Call image understanding tool
                image_context = analyze_medical_image.run(tmp_path)
                st.session_state.image_context = image_context
                os.unlink(tmp_path)

                st.success("✅ Image processed. You can now ask questions based on it.")
                st.markdown("**📝 Extracted Summary:**")
                st.write(image_context)

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": f"🖼️ Here's what I found from the image:\n\n{image_context}"}
                )
            except Exception as e:
                st.error(f"⚠ Failed to process image. Error: {e}")

    # Chat display
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    # Chat input
    user_input = st.chat_input("💬 Ask a question about the image or your health...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🤖 MediBot is thinking..."):
                try:
                    full_prompt = ""
                    if st.session_state.image_context:
                        full_prompt += f"Here is context from the medical image:\n{st.session_state.image_context}\n\n"
                    full_prompt += f"User asked: {user_input}"
                    response = run_medibot(full_prompt)
                except Exception as e:
                    response = f"⚠ Something went wrong. Error: {e}"

                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

    st.markdown("---")
    st.markdown("🧬 *Built with LangChain, OpenAI, and Streamlit*")

if __name__ == "__main__":
    gui() 