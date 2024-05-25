from flask import Flask, request, jsonify
from transformers import pipeline
from flask_ngrok import run_with_ngrok

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
# C:\Users\amit8\AppData\Local/ngrok/ngrok.yml

app = Flask(__name__)

model_name_HuggingEmbedding = "BAAI/bge-small-en-v1.5"
model_name = "theBloke/Mistral-7B-Instruct-v0.2-GPTQ"
model_classifier = "SamLowe/roberta-base-go_emotions"
top_k =3

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# Settings.embed_model = HuggingFaceEmbedding(model_name= model_name_HuggingEmbedding)
# Settings.llm = None
# Settings.chunk_size = 256
# Settings.chunk_overlap =25
# Load documents and create index
# documents = SimpleDirectoryReader("data").load_data()
# index = VectorStoreIndex.from_documents(documents)
# retriever = VectorIndexRetriever(
#     index = index,
#     similarity_top_k=top_k
# )
# query_engine = RetrieverQueryEngine(
#     retriever = retriever,
#     node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
# )
# model = AutoModelForCausalLM.from_pretrained( model_name,
                                             # device_map="auto",
                                             # trust_remote_code=False,
                                             # revision="main")

# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
# model.eval()

# def process_query(comment):
#     query = "who gets below 10 in the final grade"
#     res = query_engine.query(query)
#     context = "Context:\n"
#     for i in range(top_k):
#         context += res.source_nodes[i].text + "\n\n"
#
#     system_prompt = """
#     You are an Instructor Assistant.
#     You have two goals:
#     Answer questions as accurately as possible based on the instructions and context provided. You can perform calculations if necessary based on the provided context.
#     Provide a work plan to improve the training of the instructor based on their questions.
#     """
#     prompt_template_w_context = lambda context, comment: f'''[INST] {system_prompt}\n
#     {context} Please respond to the following comment. Use the context above if it is helpful
#     {comment} [/INST]'''
#
#     prompt = prompt_template_w_context(context, comment)
#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)
#
#     return tokenizer.batch_decode(outputs)[0]


# @app.route('/chatBot', methods=['POST'])
# def handle_query():
#     data = request.json
#     comment = data['comment']
#     response = process_query(comment)
#     return jsonify({'response': response})


@app.route('/sentimental', methods=['GET'])
def get_sentimental_text():
    data = request.get_json()
    sentences = data.get('text', [])
    results = []
    for sent in sentences:
        result = classifier(sent)
        results.append(result)
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
