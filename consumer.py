from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import LlamaCppEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from constants import CHROMA_SETTINGS, QUEUE_CHANNEL_NAME
import os
import redis
import json 

load_dotenv()

llama_embeddings_model = os.environ.get("LLAMA_EMBEDDINGS_MODEL")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')

red = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)

def main():
    llama = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)
    db = Chroma(persist_directory=persist_directory, embedding_function=llama, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    # Prepare the LLM
    callbacks = [StreamingStdOutCallbackHandler()]
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        case _default:
            print(f"Model {model_type} not supported!")
            exit;
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    # Interactive questions and answers
    while True:
        sub = red.pubsub()    
        sub.subscribe(QUEUE_CHANNEL_NAME)
        print("waiting!!:")
        for message in sub.listen():    
            if message is not None and isinstance(message, dict):    
                #print(message['data'])
                messageData = message['data']
                print(messageData)
                print(type(messageData))
                if isinstance(messageData, str):
                    data=json.loads(messageData)
                    query=data['qry']
                    print(query)
                    res = qa(query)    
                    answer, docs = res['result'], res['source_documents']
                    print(answer)
                    uuid = data["uuid"]
                    red.hset("uuid", uuid, answer)
                print("ok")

if __name__ == "__main__":
    main()
