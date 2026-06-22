Librerías que se deben instalar:
----------------------------------


sudo apt install sqlite \
pip install vllm \
pip install sentence_transformers \
pip install pypdf \
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install chainlit
------------------------------------------

1º Cargar modelo en local con VLLM
2º Ejecutar aplicación: 

 chainlit run tfgLLM.py --port 8001
