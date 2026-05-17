from openai import OpenAI
from sentence_transformers import SentenceTransformer
import sqlite3
import pypdf
import numpy
import json
import re 
import random
import sys
#from fastapi import FastAPI


from datetime import datetime, timedelta

#import itertools
#import tensorflow as tf
#import tensorflow_hub as hub


############ variables globales ################################

### llm ###
client = OpenAI(
    base_url="http://localhost:8000/v1",
   # api_key="dummy"  Si no pongo api key
        api_key="claveSegura"
)


####   conexión a la BD ####
dbConnection = sqlite3.connect('sqlite3DB.db')
cursor = dbConnection.cursor()



#preguntaIRT = [] 
usuarioId = -1

llms = {
"1": "Qwen/Qwen3-0.6B", #versión ligera y simple
"2": "Qwen/Qwen2.5-7B-Instruct" #versión más potente
}


if(len(sys.argv) >= 2):
	print ("LLM escogido: "+sys.argv[1])
	llmName = llms[sys.argv[1]]
else: 
	llmName = llms["1"] #versión por defecto, la de menos consumo.


################################ endpoints ###############################

#app = FastAPI()
#@app.post("/preguntar")
#async  def preguntar():
#	return {"prueba": "primera prueba endpoints python"}





##############  funciones auxiliares ##################################




def limpiar_respuesta_json(respuesta):
    # Algunas respuestas que genera el llm deben darse en formato JSON, debido a las alucionaciones de la IA, a veces envuelve la respuesta con carácteres de markdown.
    respuesta = respuesta.replace("```json", "").replace("```", "").strip()
    
    # extraer solo el JSON
    match = re.search(r'\{.*\}', respuesta, re.DOTALL)
    if match:
        return match.group(0)
    return 
	


def encontrarTemaId(temaNombre):
	query = "SELECT id FROM Temarios WHERE nombre = ?"
	cursor.execute(query, (temaNombre,))
	result = cursor.fetchone()
	if result:
		return result[0]
	return None

def encontrarTemaNombre(temaId):
	query = "SELECT nombre FROM Temarios WHERE id = ?"
	cursor.execute(query, (temaId,))
	result = cursor.fetchone()
	if result:
		return result[0]
	return None

######################  LLM  ####################################

def sendMessage(message, GuardarHistorial, historial = "" ):
	n = 5

	print("Mensaje a enviar al LLM: "+message)
	if(len(historial) >= n):
		#Para evitar superar el max_tokens, se borran mensajes antiguos del historial. 
		auxHistorial = historial[0]
		historial[:] = [auxHistorial] + historial[3:]
		#print("CAMBIOS EN HISTORIAL!!!__->>", len(historial))
		#for h in historial:
		#	print("#####--->>>", h)
	if GuardarHistorial: 
		#if len(contexto) != 0 :
		#	historial.append(  {"role":"user", "content" : "CONTEXTO A USAR PARA LA SIGUIENTE PREGUNTA:"+contexto } )
	
			
		historial.append( {"role":"user", "content" : message} )
		resp = client.chat.completions.create(
			model=llmName,
			messages = historial, ##REVISAR AQUÍ <------------------------------------------------------
			temperature=0.3,
			max_tokens=2048
			
		)
	
	else: 
		resp = client.chat.completions.create(
			#model="Qwen/Qwen3-0.6B",
			model=llmName,
			messages =  [{"role": "user", "content": message}],
			temperature=0.3,
			max_tokens=2048
		)
	#print("RESPUESTA RAW:", resp)
	respuestaMssg = resp.choices[0].message.content
	respuestaMssg = respuestaMssg.split("</think>").pop().strip()  #Divido string en dos partes (pensamiento de la IA y su respuesta), me quedo solo la última y elimino espacios, saltos de línea etc
	if GuardarHistorial:
		historial.append({"role": "assistant", "content": respuestaMssg})
	#print(respuestaMssg)
	return respuestaMssg




def selectorTemas(message): #Seleccionar tema más probable
	
	#listaPalabras = message.split(" ") #dividir en una lista de palabras la petición
	#n = 0
	#for palabra in listaPalabras: 
	#	if (palabra.isdigit() and n-1>= 0 and listaPalabras[n-1] == "tema" ):
	#		print("Consigo que entre aquí ¿¿??")
	#		temaEscogido = "tema "+palabra+" ppss"
	#		n += 1
	#		return temaEscogido
	query = "select nombre, descripcion from Temarios;"
	cursor.execute(query)
	datos = cursor.fetchall() #lista de tuplas
	listaOrdenada = []
	for nombre, descripcion in datos :
		listaOrdenada.append(f"{nombre} : {descripcion}")

	temas = '\n'.join(listaOrdenada) #convertir a string y añadir un salto de línea entre temas
	try:
		with open("prompts/promptEscogerTema.txt", "r", encoding="utf-8") as f:
			plantilla = f.read()
			promptEscogerTema = plantilla.replace("{ListaTemas}", temas).replace("{Peticion}" ,message )
		
	except FileNotFoundError:
		print( "Error: archivo con el prompt no encontrado. No es posible generar la pregunta")
		return
	except Exception as e:
		print( "Error: "+str(e))
		return

	respuestaIA = sendMessage(promptEscogerTema, False)
	#print("respuesta obtenida es: "+respuestaIA)
	temaEscogido = respuestaIA.split("</think>").pop().strip() 
	return temaEscogido

######################################################### RESUMIR #####################################################################################


def generarResumenes(consulta, historial):

	tema = selectorTemas(consulta)
	print("tema seleccionado: ",tema)
	datosPdf = obtenerDatosTema(tema) 
	try:
		with open("prompts/promptResumir.txt", "r", encoding="utf-8") as f:
			plantilla = f.read()
			promptResumir = plantilla.replace("{Consulta}", consulta).replace("{datosPdf}",datosPdf)
		
	except FileNotFoundError:
		print( "Error: archivo con el prompt no encontrado. No es posible generar la pregunta")
		return

	except Exception as e:
		print( "Error: "+str(e))
		return


	respuesta = sendMessage(promptResumir,False,historial)
	return respuesta




def obtenerDatosTema(tema):

	#devuelve el tema pdf entero, sin acortarlo con RAG, necesario para hacer resúmenes.	
	temaId = encontrarTemaId(tema)
	query = "select Texto from embeddings WHERE TemaId = ? ;"
	cursor.execute(query,(temaId, ) ) #aunque solo haya un elemento se debe mantener puesta la coma de tema
	chunks = cursor.fetchall()
	datosPdf = ""
	for chunk in chunks:
		datosPdf = datosPdf + chunk[0]
	return datosPdf
	
 

########################### RAG ##########################################################################################

modelEmb = SentenceTransformer('all-MiniLM-L6-v2')
UMBRAL = 0.5 #similitud míinima requerida
#Retrieval -> semantic search
#Augmented knowledge-> Inject into promt
#Generation 

def buscarSimilitud(message): #Retrieval. -> RAG, lee embeddings del tema seleccionado en el primer nivel
	messageEmbedding = modelEmb.encode(message) #transformar mensaje a secuencia de números, gracias a un modelo externo
	query = "select * from embeddings WHERE TemaId = ? ;"
	tema = selectorTemas(message)
	temaId = encontrarTemaId(tema)
	#print("tema seleccionado: ",tema)
	cursor.execute(query,(temaId, ) )
	chunks = cursor.fetchall()
	similarities = []
	aux = [] #lista auxiliar, realiza copia de los textos en orden para poder ampliar el contexto en siguientes pasos.
	for chunk in chunks:
	
		embedding = numpy.frombuffer(chunk[3], dtype=numpy.float32)
		similarity  = cosine_similarity(messageEmbedding, embedding)
		if similarity >= UMBRAL:
			similarities.append((similarity,chunk[2]))

		aux.append(chunk[2])
	similarities.sort(key=lambda x: x[0], reverse=True)
	response = []
	topN = 0 #cambiar nombre a topk
	plusN = 6
	
	if len(similarities) >= 6 :
		topN =6

	else:
		topN = len(similarities)
	
	print("topN es", topN)

	for s in similarities[:topN]: #Guardamos las mejores similitudes y ampliamos su contexto

		iterator = aux.index(s[1])  #posición en aux del elemento 
	
		#Para más contexto  añadimos las plusN - 1 líneas anteriores y siguientes a los chunks más probables de ser lo que buscamos
		for i in range (1,plusN):
	
			if iterator - (plusN - i) >= 0:
				response.append(aux[iterator - (plusN - i) ])

		response.append(s[1])
		for i in range(1, plusN):
			if iterator + i < len(aux):
				response.append(aux[iterator + i])		
	return response

def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)



######## Generar nuevos embedings de los pdf almacenados #####################################################################################
def generarEmbedingsTemas():
	# 0. Borrar datos previos
	# 1. Recorro todos los pdf, y divido cada una de las páginas de los pdf en  líneas (chunks)
	# 2. Generar embeding en base a modelo entrenado
	# 3. Almacenar nuevos datos
	queryBorrar ="DELETE FROM embeddings;"
	cursor.execute(queryBorrar)
	dbConnection.commit()
	
	
	query = "SELECT ID,path FROM Temarios;"
	cursor.execute(query)
	responses = cursor.fetchall()
	temas = []
	for res in responses:
		pdfContent = pypdf.PdfReader(res[1]) #leer pdf
		temaId = res[0]
		chunks = []
		for pdfPage in pdfContent.pages: #obtener chunks, página por página del pdf
			contenido = pdfPage.extract_text()
			if contenido:
				partes = contenido.split("\n")
				i = 0
				for p in partes:
					i += 1
					chunks.append(p)
			
		
		embeddings = modelEmb.encode(chunks) #generar embedding con el modelo
		temas.append((temaId,chunks,embeddings))

	print("---->temas: ", temas)

		#command = "Insert into embeddings(Nombre, Embedding) VALUES ('" , chunk ,"' ,'", embeddings[i],")"
	for tema in temas:
		for i, chunk in enumerate(tema[1]) : #tema[1] = chunks. Hay el mismo número de embeddings que de chunks.
			command = "INSERT INTO embeddings (TemaId, Texto , Embedding) VALUES (?, ?, ?)"
			cursor.execute(command, (tema[0], chunk, tema[2][i].astype(numpy.float32).tobytes())) #cambio tipo del embedding para almacenarlo en sqlite
			dbConnection.commit()



############################## PREGUNTAR ###############################################################################################

def generarPregunta(consulta, historial): #En lugar de usar retrieval, se devuelve el temario seleccionado completo.
	

	temaPregunta = selectorTemas(consulta)
	print("tema: "+temaPregunta)
	print("tema pregunta: "+temaPregunta)
	temaId = encontrarTemaId(temaPregunta)
	query = "Select NivelConocimiento from Conocimientos WHERE UsuarioId = ? AND TemaId = ?"
	cursor.execute(query, (usuarioId, temaId))
	datos = cursor.fetchall()
	if(len(datos) >  0):
			conocimientoAlumno = datos[0][0]
	else:
			conocimientoAlumno = 0

	datosPdf = obtenerDatosTema(temaPregunta)

	try:
		with open("prompts/promptPreguntar.txt", "r", encoding="utf-8") as f:
			plantilla = f.read()
			promptPreguntar = plantilla.replace("{Consulta}", consulta).replace("{datosPdf}" , datosPdf).replace("{conocimientoAlumno}",str(conocimientoAlumno))
		
	except FileNotFoundError:
		print( "Error: archivo con el prompt no encontrado. No es posible generar la pregunta")
		return

	except Exception as e:
		print( "Error: "+str(e))
		return

	respuesta = sendMessage(promptPreguntar,False,historial).strip()
	print("RAW:"+respuesta)

	jsonLimpio = limpiar_respuesta_json(respuesta)
	print(jsonLimpio)
	if not jsonLimpio:
		#raise ValueError("No se pudo extraer JSON")
		print( "Error: no se pudo  generar correctamente su pregunta")
		return
	
	data = json.loads(jsonLimpio)
	preguntaIRT = []
	preguntaIRT.append(data["pregunta"].strip())
	preguntaIRT.append(data["a"])
	preguntaIRT.append(data["b"])
	preguntaIRT.append(temaPregunta)
	return preguntaIRT



def cuestionario(consulta,historial):
	## 1. Calcular número de preguntas a realizar al usuario.
	## 2. Generar las preguntas.
	## 3. Comprobar respuestas y actualizar el conocimiento que el sistema tiene sobre el alumno.
	aciertos = 0
	numPreguntas = -1
	n = 0
	listaPalabras = consulta.split(" ")
	for palabra in listaPalabras: 
		if palabra.isdigit() and n+1 < len(listaPalabras)  and listaPalabras[n+1] == "preguntas" or (palabra == "1")  and n+1 < len(listaPalabras) and listaPalabras[n+1] == "pregunta"  :
			numPreguntas = int(palabra)
		n += 1
	if( numPreguntas == -1):
		try:
			with open("prompts/promptNumeroPreguntas.txt", "r", encoding="utf-8") as f:
				plantilla = f.read()
				promptNumeroPreguntas = plantilla.replace("{Consulta}",consulta)
		
		except FileNotFoundError:
			print( "Error: archivo con el prompt no encontrado. No es posible generar la pregunta")
			return
		except Exception as e:		
			print( "Error: "+str(e))
			return
		
		response = sendMessage(promptNumeroPreguntas,False)
		
		if(response.isdigit()):
			numPreguntas = int(response)
		else:
			print("Error: no se pudo calcular el número de preguntas")
			return
	
	print("numPreguntas: "+ str(numPreguntas))
	for i in range(0,numPreguntas) :

			
		if ("desarrollo" in listaPalabras):
			tipoPregunta = "desarrollo"
		elif ("test" in listaPalabras or "tests" in listaPalabras):
			tipoPregunta = "test"
		else: 
			tipoPregunta = random.choice(["desarrollo", "test"])


		if tipoPregunta == "desarrollo":
			preguntaIRT = generarPregunta(consulta,historial)

			#array con los datos: 0. pregunta , 1. valor a ,2. valor b, 3. tema pregunta
			print(preguntaIRT[0])
			respuesta = input()
		
			solucion = evaluarRespuesta(respuesta,historial, preguntaIRT)
			print(solucion)

			
		elif tipoPregunta == "test":
			solucion = generarPreguntaTest(consulta, historial)

		listaSolucion = solucion.split(" ")
		print("soluencuestionario: " +str(listaSolucion))
		if listaSolucion[0] == "CORRECTO:":
			aciertos += 1
	print("-----------------------------")
	print ("Preguntas acertadas:"+ str(aciertos)+"/"+str(numPreguntas))
	print("-----------------------------")



############################# Evaluar conocimiento alumno ####################################################################################


def evaluarRespuesta(consulta,historial, preguntaIRT):

	try:
		with open("prompts/promptVerificarRespuesta.txt", "r", encoding="utf-8") as f:
			plantilla = f.read()
			promptVerificarRespuesta = plantilla.replace("{Pregunta}",preguntaIRT[0]).replace("{Respuesta}", consulta)
		
	except FileNotFoundError:
		print ("Error: archivo con el prompt no encontrado. No es posible generar la pregunta")
		return

	except Exception as e:
		print( "Error: "+str(e))
		return

	response = sendMessage(promptVerificarRespuesta,False,historial)
	jsonLimpio = limpiar_respuesta_json(response)
	data = json.loads(jsonLimpio)
	evaluacion = data["Evaluacion"]
	motivo = data["Motivo"]
	solucion = data["Solucion"]
	res = evaluacion +": "+motivo+ " "+solucion

	query = "INSERT INTO PREGUNTAS (Pregunta,TemaId,Tipo,Solucion,a,b) VALUES ( ? , ? ,? ,?, ?, ?);"
	temaId = encontrarTemaId(preguntaIRT[3]) #con el nombre del tema busco  su id
	cursor.execute(query, (preguntaIRT[0],temaId,"Desarrollo",solucion, preguntaIRT[1], preguntaIRT[2]))
	dbConnection.commit()
	preguntaId = cursor.lastrowid
	if( evaluacion == "CORRECTO"):
		acierto = True
	elif(evaluacion == "FALSO"):
		acierto = False
	else:
		return "Error: Fallo durante la evaluación de la respuesta"
	
	query =  "INSERT INTO PreguntasUsuarios (PreguntaId,UsuarioId, Acierto) VALUES (?,? , ?) "
	cursor.execute(query,(preguntaId,usuarioId,acierto))
	dbConnection.commit()

	actualizarNivelCononocimientoAlumno(evaluacion, preguntaIRT, temaId)
	return res 


def actualizarNivelCononocimientoAlumno(evaluacion,preguntaIRT,temaId, preguntaTest = False, confianza = 0):


	if evaluacion == "CORRECTO":
		resultado = 1
	elif evaluacion == "FALSO":
		resultado = 0
		###GUARDAR PREGUNTA TODO
	else:
		print("Error al evaluar su respuesta.")
		return

	consulta = "Select NivelConocimiento from Conocimientos WHERE UsuarioId = ? AND TemaId = ?"
	cursor.execute(consulta, (usuarioId, temaId))
	datos = cursor.fetchall()
	if(len(datos) >  0):
			conocimientoAlumno = datos[0][0]
	else:
			conocimientoAlumno = 0
	
	probAcierto  = irt(conocimientoAlumno, preguntaIRT,preguntaTest)
	a = float(preguntaIRT[1])
	 #Tasa de aprendizaje o "peso" de la pregunta. Cuanto afecta al conocimiento. Una pregunta repetida afecta menos.
	if preguntaTest:

		k_confianza = {
        	1: 0.25,   # acierto/fallo con poca confianza → actualización suave
        	2: 0.5,   # confianza media → actualización normal
        	3: 0.8,   # alta confianza → actualización fuerte
   		}
		k = k_confianza[confianza]
	else:
		k = 0.5

			
		
	nuevoValorConocimiento = conocimientoAlumno + k*a*( resultado - probAcierto)

	#Si es la primera vez que se crea un registro del nivel del alumno para el tema ,hacemos INSERT, si no, hacemos UPDATE
	query = """ INSERT INTO Conocimientos (UsuarioId, TemaId, NivelConocimiento)
				VALUES (?, ?, ?)
				ON CONFLICT(UsuarioId, TemaId)
				DO UPDATE SET NivelConocimiento =  ?
			"""

	cursor.execute(query, (usuarioId, temaId, nuevoValorConocimiento, nuevoValorConocimiento))
	dbConnection.commit()


def irt(conocimientoAlumno, preguntaIRT,preguntaTest): ##fórmula de evaluación
	#a -> discriminación. Que tan bien distinque entre alumnos buenos y malos. Cuanto más a (más pendiente en la curva) es más probable que solo los alumnos de alto nivel acierten
	#b -> dificultad de la pregunta.
	#O -> Nivel del alumno para un tema concreto.
	#P(X=1) probabilidad de acertar. (X=0 prob. fallar no relevante)
	#P(x=1) = 1/(1+e^-a(O-b))

	numE = 2.71828
	a = float(preguntaIRT[1])
	b = float(preguntaIRT[2])

	if(preguntaTest): #3PL
		c = 0.25 #tasa de acierto por azar. preguntas tipo test 4 opciones -> 0,25
	else: #2PL
		c = 0 #preguntas de desarrollo, azar despreciable 
	
	probAcierto =  c + (1 - c)/(1+numE**(-a*(conocimientoAlumno-b)))
	return probAcierto

######################################### PREGUNTAS TIPO TEST ############################################


def nivel_bloom(theta_conocimiento):
	#Conforme un alumno avance en un tema  las preguntas irán subiendo al siguiente nivel dela taxonomía de Bloom.
    if theta_conocimiento < 0.5:
        return 1  # Recordar — inicio del tema
    elif theta_conocimiento < 1.5:
        return 2  # Comprender
    elif theta_conocimiento < 2.5:
        return 3  # Aplicar
    else:
        return 4  # Analizar





def generarPreguntaTest(consulta, historial):
	
	temaPregunta = selectorTemas(consulta)
	print("tema pregunta: "+temaPregunta)
	temaId = encontrarTemaId(temaPregunta)
	query = "Select NivelConocimiento from Conocimientos WHERE UsuarioId = ? AND TemaId = ?"
	cursor.execute(query, (usuarioId, temaId))
	datos = cursor.fetchall()
	
	if(len(datos) >  0):
			conocimientoAlumno = datos[0][0]
	else:
			conocimientoAlumno = 0

	datosPdf = obtenerDatosTema(temaPregunta)
	try:
		with open("prompts/promptTest.txt", "r", encoding="utf-8") as f: #FALTA AÑADIR CONOCIMIENTO AL PROMPT !!!!

			plantilla = f.read()
			nivelBloom = nivel_bloom(conocimientoAlumno)
			promptTest = plantilla.replace("{Consulta}", consulta).replace("{datosPdf}" , datosPdf).replace("{conocimiento}",str(conocimientoAlumno)).replace("{nivelBloom}", str(nivelBloom))


	except FileNotFoundError:
		print( "Error: archivo con el prompt no encontrado. No es posible generar la pregunta")
		return
	except Exception as e:		
		print( "Error: "+str(e))
		return
	
	respuesta = sendMessage(promptTest,False, historial)
	jsonLimpio = limpiar_respuesta_json(respuesta)
	data = json.loads(jsonLimpio)
	preguntaTest = []
	preguntaTest.append(data["Pregunta"].strip())
	preguntaTest.append(data["OpcionA"])
	preguntaTest.append(data["OpcionB"])
	preguntaTest.append(data["OpcionC"])
	preguntaTest.append(data["OpcionD"])
	preguntaTest.append(data["Solucion"])
	preguntaTest.append(temaPregunta)
	preguntaTest.append(data["a"])
	preguntaTest.append(data["b"])
	return mostrarPreguntaTest(preguntaTest)



def mostrarPreguntaTest(preguntaTest):
	
	print("Pregunta: "+preguntaTest[0])
	print("A: "+preguntaTest[1])
	print("B: "+preguntaTest[2])
	print("C: "+preguntaTest[3])
	print("D: "+preguntaTest[4])
	temaPregunta = preguntaTest[6]
	a = preguntaTest[7]
	b = preguntaTest[8]
	alumnoRespuesta = ""
	while alumnoRespuesta != "A" and alumnoRespuesta != "B" and alumnoRespuesta != "C" and alumnoRespuesta != "D":
		print("Escriba la letra de la opción que considere correcta: ")
		alumnoRespuesta = input().strip().upper()

		if alumnoRespuesta != "A" and alumnoRespuesta != "B" and alumnoRespuesta != "C" and alumnoRespuesta != "D":
			print("Respuesta no válida. Por favor, escriba A, B, C o D.")

	nivelSeguridad = ""
	while nivelSeguridad != "1" and nivelSeguridad != "2" and nivelSeguridad != "3":
		print("¿ Cómo de seguro está de su respuesta? No está seguro (1), seguro a medias (2), bastante seguro (3)")
		nivelSeguridad = input()
		if nivelSeguridad != "1" and nivelSeguridad != "2" and nivelSeguridad != "3":
			print("Respuesta no válida. Por favor, escriba 1, 2 o 3.")


	nivelSeguridad = int(nivelSeguridad)
	solucion = preguntaTest[5].strip().upper()

	if(alumnoRespuesta == solucion):
		
		print("¡Respuesta correcta!")
		evaluacion = "CORRECTO"
	else:
		evaluacion = "FALSO"
		print("Respuesta incorrecta. La respuesta correcta es: "+solucion)
	

	query = "INSERT INTO PREGUNTAS (Pregunta,TemaId,Tipo,Solucion,a,b) VALUES ( ? , ? ,? ,?, ?, ?);"
	temaId = encontrarTemaId(temaPregunta) 
	cursor.execute(query, (preguntaTest[0],temaId,"Test",solucion, nivelSeguridad, 0))
	dbConnection.commit()

	query = "INSERT INTO PreguntasUsuarios (PreguntaId,UsuarioId, Acierto) VALUES (?,? , ?) "
	cursor.execute(query,(cursor.lastrowid,usuarioId,evaluacion == "CORRECTO"))
	dbConnection.commit()

	preguntaIRT = [preguntaTest[0], a, b, temaPregunta] #preguntas tipo test, a = 1 , b = 0
	actualizarNivelCononocimientoAlumno(evaluacion,preguntaIRT, temaPregunta, True, nivelSeguridad) 
	return evaluacion



########################## Log in ##########################################################################



def login():
	global usuarioId 
	print("Introduzca su usuario: (para registrar un nuevo usuario pulse la tecla 0)")
	usuario = input()

	if(usuario == "0"): 
		res = registrar()
		return res
	print("Introduzca su contraseña: ")
	contra = input()

	consulta = "Select id from Usuarios WHERE Nombre = ? AND Contrasena = ?"
	cursor.execute(consulta, (usuario,contra))
	datos = cursor.fetchall()

	if len(datos) == 0:
		#Usuario no existe
		print("Error: Usuario no registrado")
		res = False
	elif len(datos) == 1:
		usuarioId = datos[0][0]
		print(usuarioId)
		res = True
	else:
		print("Error de integridad en la BD")
		res = False
	return res


def registrar():
	global usuarioId
	print("Introduzca un nombre para su usuario: ")
	usuario = input()
	print("Introduzca una contraseña: ")
	contra = input()

	query = "INSERT INTO Usuarios (Nombre, Contrasena) VALUES (?,?) "

	try:
		cursor.execute(query, (usuario, contra))
		dbConnection.commit()
		usuarioId = cursor.lastrowid
		return True
	except sqlite3.IntegrityError:
		print("ERRROR: Usuario ya existe")
		return False
	except Exception as ex:
		print("ERROR: "+str(ex))


def isLoged(): #true/false si hay usuario logeado

	if usuarioId != -1 :
		return True
	else:
		return False
	

########################################### Evento repaso ################################################


def pasanDosSemanas(hoy,origen):

	dias = hoy - origen
	if( dias.days%14 == 0) : #cada dos semanas 
		return  True
	else:
		return False


def seleccionarTemasARecordar(historial) : 

	query = "SELECT nombre,fecha FROM Temarios;"
	cursor.execute(query)
	datos = cursor.fetchall() #lista de tuplas
	temasRecordar = []
	hoy =  datetime.now().date()
	hoy = datetime.strptime("2020/4/13", '%Y/%m/%d').date()

	for dato in datos: 

		if len(dato) != 2:
			print("Error grave")
		nombreTema = dato[0]
		fechaTema = dato[1] #Fecha de finalización de un tema en string
		fechaTema = datetime.strptime(fechaTema, '%Y/%m/%d').date() #convertir a tipo Date
		
		#if( hoy - fechaTema == timedelta(days =14)): 
		if (pasanDosSemanas(hoy,fechaTema)):
			temasRecordar.append(nombreTema)
			semanas = (hoy - fechaTema)/14


	if( len(temasRecordar) >0 ):
		eleccionUsuario = ""
		while( eleccionUsuario != "s" and eleccionUsuario != "n" ):	
			print("Hoy, se cumplen "+ str(semanas.days) + " semanas desde que en clase se terminó de estudiar los temas: " + str(temasRecordar) + " . ¿Desea repasar estos temarios?. (Escriba S o N)")
			eleccionUsuario = input().lower()
			if( eleccionUsuario != "s" and eleccionUsuario != "n" ):
				print("Escriba S o N")
			elif(eleccionUsuario == "s"):
				recordarTemas(temasRecordar,historial)

			
	
def recordarTemas( temasRecordar,historial) :

	#Se hace al usuario una bateria de las preguntas que ha hecho anteriormente sobre un tema/temas concreto
	query = "SELECT p.Pregunta, p.Solucion, p.a, p.b, p.ID FROM Preguntas p JOIN PreguntasUsuarios pu ON pu.PreguntaId = p.id WHERE p.TemaId = ? AND pu.UsuarioId = ? ;"
	for tema in temasRecordar:
		temaId = encontrarTemaId(tema)
		cursor.execute(query, (temaId,usuarioId ))
		datos = cursor.fetchall() #lista de tuplas 
		if(len(datos) == 0):
			print("No hay preguntas a repasar por ahora sobre el tema "+tema)
		else: 
			aciertos = 0
			for dato in datos:	
				if len(dato) != 5:
					print("Error grave")
					return
				preguntaIRT = [ dato[0], dato[2], dato[3], tema]
				respuestaEsperada = dato[1]
				preguntaId = dato[4]
				print(dato[0])
				respuestaAlumno = input()
				
				
				alumnoAcierta = compararSoluciones(preguntaIRT, respuestaEsperada, respuestaAlumno,historial)

				if alumnoAcierta : 
					aciertos += 1
				else:
					print(respuestaEsperada)

				queryActualizar = "UPDATE PreguntasUsuarios SET Acierto = ? WHERE UsuarioId = ? AND PreguntaId = ?"
				cursor.execute(queryActualizar, (alumnoAcierta, usuarioId, preguntaId))
				dbConnection.commit()
			
			print("------------------------------------")
			print("Evaluación del repaso del tema "+tema)
			print("------------------------------------")
			print("Has acertado: "+str(aciertos) + " / "+str(len(datos)))
			print("------------------------------------")
	




def compararSoluciones(preguntaIRT, respuestaEsperada, respuestaAlumno,historial):

	try:
		with open("prompts/promptCompararRespuestas.txt", "r", encoding="utf-8") as f:
			plantilla = f.read()
			promptCompararRespuestas = plantilla.replace("{Pregunta}",preguntaIRT[0]).replace("{respuestaAlumno}", respuestaAlumno).replace("{respuestaEsperada}", respuestaEsperada)
		
	except FileNotFoundError:
		print( "Error: archivo con el prompt no encontrado. No es posible generar la pregunta")
		return
	except Exception as e:		
		print( "Error: "+str(e))
		return
	respuesta = sendMessage(promptCompararRespuestas,False, historial)
	jsonLimpio = limpiar_respuesta_json(respuesta)
	data = json.loads(jsonLimpio)
	evaluacion = data["Evaluacion"]
	print(evaluacion)
	temaId = encontrarTemaId (preguntaIRT[3]) 

	#queryPeguntasUsuarios = "SELECT Acierto FROM PreguntasUsuarios WHERE UsuarioId = ? AND PreguntaId = ?"
	#cursor.execute(queryPeguntasUsuarios, (usuarioId, preguntaId))
	#datosPreguntaUsuario = cursor.fetchall()[0]
	#print("anterior---->>"+str(datosPreguntaUsuario))

	actualizarNivelCononocimientoAlumno(evaluacion,preguntaIRT,temaId, False )
	if(evaluacion == "CORRECTO"):
		return True #pregunta acertada
	else:
		return False #Pregunta fallada





############################ seguimiento alumno #####################################################################

def seguimientoAlumno():

	querySelect = "SELECT TemaId,NivelConocimiento FROM Conocimientos where usuarioId = ?"

	cursor.execute(querySelect, (usuarioId, ))
	asignaturas = cursor.fetchall() #lista de tuplas 
	if(len(asignaturas) == 0):
		print("No hay registros sobre el conocimiento del usuario. Practique, pidiendo a Qwen que le haga preguntas sobre el tema que quiera repasar.")
	else:

		print("Estado de los temas que ha practicado: ")
		print("---------------------------------------")
	for asinatura in asignaturas:
		print("--->",  asinatura)
		temaId = asinatura[0]
		tema = encontrarTemaNombre(temaId)
		conocimiento = asinatura[1]
		#ToDo: Buscar valores para cambiar el 0.8
		if (conocimiento <= -0.8):
			print(tema+": El usuario ha fallado muchas preguntas, conocimientos muy bajos sobre el tema. Se aconseja estudiar el tema. Conocimiento registrado actual: "+str(conocimiento))
		elif(conocimiento < 0 and conocimiento > -0.8 ):
			print(tema+": Ligeramente por debajo del promedio, se aconseja repasar el tema.  Conocimiento registrado actual: "+str(conocimiento))
		elif( conocimiento >= 0 and conocimiento <= 0.8):
			print(tema+": Ligeramente por encima del promedio, se aconseja seguir practicando el tema. Conocimiento registrado actual: "+str(conocimiento))
		else:
			print(tema+": El usuario ha acertado muchas preguntas sobre el tema."+str(conocimiento))
		
		queryPreguntasContar = "SELECT COUNT(*) FROM PreguntasUsuarios WHERE UsuarioId = ? AND Acierto = True"
		cursor.execute(queryPreguntasContar, (usuarioId, ))
		numPreguntasAcertadas = cursor.fetchall()[0][0]

		queryPreguntasContar = "SELECT COUNT(*) FROM PreguntasUsuarios WHERE UsuarioId = ? AND Acierto = False"
		cursor.execute(queryPreguntasContar, (usuarioId, ))
		numPreguntasFalladas = cursor.fetchall()[0][0]
		print("Número de preguntas acertadas sobre el tema: "+str(numPreguntasAcertadas))
		print("Número de preguntas falladas sobre el tema: "+str(numPreguntasFalladas))
		print("---------------------------------------")






####################################### MAIN ############################################################################







def router(consulta,historial): #función encargada de redirigir a donde se debe tratar el mensaje, según su contenido.
	response = ""
	consulta = consulta.lower() #quitar mayúsculas para poder simplificar la detección de palabras clave del router
		#si se detecta alguna estructura que sea "HAZME UNA PREGUNTA" , "PREGÚNTAME" ... , no es necesario usar el prompt, es obvio lo que quiere el usuario.
	palabrasClave = ["haz","genera","crea","inventa"]
	if ( any(palabra in consulta for palabra in palabrasClave) and ("pregunta" in consulta)) or ("preguntame" in consulta):
		tipoMensaje = "RAG_EXAM"
	elif ( any(palabra in consulta for palabra in palabrasClave) and ("resumen" in consulta)) or ("resumeme" in consulta) :
		tipoMensaje = "RAG_RESUME"
	else: #analizar que camino escoger con el prompt previo
		try:
			with open("prompts/promptRouter.txt", "r", encoding="utf-8") as f:
				plantilla = f.read()
				promptRouter = plantilla.replace("{Consulta}", consulta)
		
		except FileNotFoundError:
			print("Error: archivo con el prompt no encontrado. No es posible generar la pregunta")
			return
		except Exception as e:
			print("Error: "+str(e))
			return 
		tipoMensaje = sendMessage(promptRouter, False)	
		tipoMensaje = tipoMensaje.split("</think>").pop()
	print("-->", tipoMensaje)
	match tipoMensaje:
		case  "NORMAL":
			response = sendMessage(consulta,True, historial)
			print(response)
		case "RAG_EXAM":
			cuestionario(consulta, historial)
		case "RAG_RESUME":
			response = generarResumenes(consulta,historial)
			print(response)
		case "RAG_INFO": 
				#Buscar contexto con RAG y responder la pregunta del usuario con la información extraida de los PDF.
				chunks = buscarSimilitud(consulta) #Retrieval de RAG
				contexto = ' '.join(chunks)		
				#Insertar contexto -> Augmented Knowledge de R.A.G
				try:
					with open("prompts/promptInformar.txt", "r", encoding="utf-8") as f:
						plantilla = f.read()
						promptInformar = plantilla.replace("{Consulta}", consulta).replace("{Contexto}", contexto)
		
				except FileNotFoundError:
					print("Error: archivo con el prompt no encontrado. No es posible generar la pregunta")
					return
				except Exception as e:
					print("Error: "+str(e))
					return 

				response = sendMessage(promptInformar,True,historial) #Generation
				print(response)

		case _:
			print("ERROR")

def chat(historial, recordarUnaVez):
	print("-------------------------------------------------------------------------------")
	print("\n")
	print("                               CHAT CON QWEN")
	print("\n")
	print("Para volver al menú pulse 0")
	print("-------------------------------------------------------------------------------")

	if recordarUnaVez :
		seleccionarTemasARecordar(historial)
		recordarUnaVez = False
		print("..............................")
	print("Bienvenido! Soy un tutor virtual inteligente, mi objetivo es ayudarte con ppss")
	print("..................................")
	consulta = input()
	print("..................................")

	while(consulta != "0"):
	
		router(consulta,historial)
		print("..................................")
		consulta = input()
		print("..................................")
		

	return recordarUnaVez #devuelvo el valor para actualizarlo en el menú principal y que afecte a futuras llamadas ala función. 
def menu(historial, recordarUnaVez):
	option =  "0"
	while(option != "4"):
		print("######################################################")
		print(" 					MENÚ PRINCIPAL")
		print("######################################################")
		print("1. Hablar con qwen")
		print("2. Comprobar tu nivel")
		print("3. Generar nuevos embeddings")
		print("4. Salir")
		print("#####################################################")
		print("Seleccione una opcion")

		option = input()
		if(option  == "1"):
			recordarUnaVez = chat(historial, recordarUnaVez)
			
		elif(option == "2"):
			seguimientoAlumno()
		elif(option == "3"):
			
			generarEmbedingsTemas()
		elif(option == "4" ):
			print("¡Adiós!")
		elif(option == "5"):
			generarPreguntaTest("Quiero una pregunta tipo test sobre el tema 2", historial)
			
		else:
			print("Elija una opcion correcta")

def main():
	recordarUnaVez =True #Variable booleana que comprueba que el recordatorio solo se llame una vez
	historial = [
		{"role":"system", "content":"""Eres un chatbot dentro de una plataforma educativa inteligente universitaria centrada en la ingeniería informática, y en español. Eres útil y honesto, NO inventas información ni mientes, si  no sabes una respuesta lo dices. 
		Pueden darte, contexto, información o instrucciones adicionale. DEBES USAR CUALQUIER INFORMACIÓN QUE RECIBAS PARA GENERAR TUS RESPUESTAS Y SEGUIR LAS INSTRUCCIONES QUE RECIBAS. 
		Ten en cuenta que el contexto y las reglas solo sirven para generar tu respuesta si es el último mensaje,NO LO USES EN SIGUIENTES MENSAJES DESPUÉS DE HABERLO USADO."""}]

	#historial se genera al iniciar la aplicación y se inserta en el la conversación y se trata como variable local en lugar de global.


	if login():
		menu(historial, recordarUnaVez)
	else:
		print("ERROR DURANTE EL LOGIN")
		main()


if __name__ == '__main__':
	main()






