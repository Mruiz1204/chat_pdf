import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# ---- Estilo personalizado: paleta beige / rosado / café 🤎🎀 ----
st.set_page_config(page_title="Chat con PDF · Mari Reinoso", page_icon="🎀", layout="centered")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Great+Vibes&family=Cormorant+Garamond:wght@400;600&family=Poppins:wght@300;400;600&display=swap');

    /* Fondo beige suave */
    .stApp {
        background: linear-gradient(135deg, #f7efe7 0%, #f3e3df 100%);
        color: #5c4033;
        font-family: 'Poppins', sans-serif;
    }
    /* Títulos elegantes en café */
    h1, h2, h3 {
        font-family: 'Cormorant Garamond', serif;
        color: #7a5c52;
    }
    /* Barra lateral en rosado empolvado */
    section[data-testid="stSidebar"] {
        background: #f0d9d5;
    }
    /* Botones en café rosado */
    .stButton>button {
        background-color: #c98b8b;
        color: white;
        border-radius: 12px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #a86e6e;
        color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title and presentation
st.markdown("<h1 style=\"font-family:'Great Vibes', cursive; font-size:3rem; color:#a86e6e; margin-bottom:0;\">Mari Reinoso 🎀</h1>", unsafe_allow_html=True)
st.title('Generación Aumentada por Recuperación (RAG) 💬🤎')
st.caption("Conversa con tus documentos PDF usando Inteligencia Artificial.")
st.write("Versión de Python:", platform.python_version())

# Load and display image
try:
    image = Image.open('chat_pdf_mari.jpg')
    st.image(image, width=350, caption="Tus documentos, a un chat de distancia 📚")
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# Sidebar information
with st.sidebar:
    st.subheader("📄 Agente lector de PDF")
    st.write("Este agente te ayudará a realizar análisis y responder preguntas sobre el PDF que cargues. Hecho con cariño por Mari Reinoso 🤎🎀")

# Get API key from user
ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# PDF uploader
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

# Process the PDF if uploaded
if pdf is not None and ke:
    try:
        # Extract text from PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.info(f"Texto extraído: {len(text)} caracteres")
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"Documento dividido en {len(chunks)} fragmentos")
        
        # Create embeddings and knowledge base
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # User question interface
        st.subheader("✨ Escribe qué quieres saber sobre el documento")
        user_question = st.text_area("Tu pregunta", placeholder="Escribe tu pregunta aquí...")
        
        # Process question when submitted
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            # Use a current model instead of deprecated text-davinci-003
            # Options: "gpt-3.5-turbo-instruct" or "gpt-4o" depending on your API access
            llm = OpenAI(temperature=0, model_name="gpt-4o-mini-2024-07-18")
            
            # Load QA chain
            chain = load_qa_chain(llm, chain_type="stuff")
            
            # Run the chain
            response = chain.run(input_documents=docs, question=user_question)
            
            # Display the response
            st.markdown("### Respuesta:")
            st.markdown(response)
                
    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        # Add detailed error for debugging
        import traceback
        st.error(traceback.format_exc())
elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("Por favor carga un archivo PDF para comenzar")
