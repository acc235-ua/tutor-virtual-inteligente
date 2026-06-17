
#Programa para calcular las métricas de BERTscore

from bert_score import score

referencia = [""" Un ARTEFACTO es cualquier fichero que Maven iden-
tifica por sus COORDENADAS, y que Maven descarga, instala o despliega de forma
automática. Ejemplos: ficheros *.pom, *.jar, *.war, *.ear
Las COORDENADAS de un artefacto se indican como: groupId:artifactId:version
groupId: es el identificador de Grupo (<groupId/>)
artifactId: es el identificador del artefacto (nombre del archivo) (<artifactId/>)
version: es la versión del artefacto (<version/>)
packaging. es el empaquetado del artefacto (<packaging/>). Por defecto su valor es
jar. No se incluye de forma explícita en las coordenadas (aunque podemos añadirlo al
final)
Los artefactos Maven residen en REPOSITORIOS:
remotos (p.ej. https://mvnrepository.com/)
locales (p.ej. $HOME/.m2/repository/)
Las coordenadas de un artefacto, no sólo identifican al fichero correspondiente, sino que
también nos permiten LOCALIZAR exactamente dónde se encuentra dicho fichero (en
un repositorio local o remoto). Por ejemplo:
coordenadas: org.junit.jupiter:junit-jupiter:5.11.4
fichero en el disco duro: $HOME/.m2/repository/org/junit/jupiter/junit-jupiter/ 5.11.4/junit-
jupiter-5.11.4.jar
Cuando maven necesita usar algún artefacto para construir el proyecto, primero com-
prueba si el ficherro correspondiente está almacenado en el repositorio local, y si no, lo
descargar del repositorio remoto"""]


candidato  = [""" Un **artefacto Maven** es cualquier fichero que
Maven identifica y maneja de forma automática. Estos ficheros pueden ser de varios ti-
pos, como ‘.pom‘, ‘.jar‘, ‘.war‘, ‘.ear‘, entre otros. Maven se encarga de descargar, instalar
y desplegar estos artefactos según sea necesario para construir y gestionar el proyecto.
Las **coordenadas de un artefacto** son una forma única de identificar cada artefacto
en Maven. Estas coordenadas se componen de tres elementos principales:
1. **groupId**: Es el identificador de grupo. Generalmente, se utiliza para representar
el dominio o el espacio de nombres del proyecto. Por ejemplo, ‘org.springframework‘
para los artefactos de Spring Framework.
2. **artifactId**: Es el identificador del artefacto, que es el nombre del archivo. Por
ejemplo, ‘spring-core‘ para el núcleo de Spring.
3. **version**: Es la versión del artefacto. Por ejemplo, ‘5.3.10‘.
Estas coordenadas se escriben en el siguiente formato: “‘ groupId:artifactId:version “‘
6.3. Encuestas a usuarios 49
Por ejemplo, el artefacto ‘spring-core‘ de la versión ‘5.3.10‘ de Spring Framework tendría
las coordenadas: “‘ org.springframework:spring-core:5.3.10 “‘
Además, hay un cuarto elemento opcional que no se incluye en las coordenadas pero
que puede ser especificado en el archivo ‘pom.xml‘:
4. **packaging**: Es el tipo de empaquetado del artefacto. Por defecto, su valor es
‘jar‘, pero también puede ser ‘war‘, ‘ear‘, entre otros. Este valor no se incluye en las
coordenadas, pero se puede especificar en el archivo ‘pom.xml‘ si es necesario.
En resumen, las coordenadas de un artefacto Maven son una forma única de identificar
y gestionar los ficheros necesarios para un proyecto, permitiendo a Maven descargarlos,
instalarlos y desplegarlos de forma automática.""" ]

P, R, F1 = score(candidato, referencia, lang="es")

print(f"Precisión: {P.mean():.4f}")
print(f"Recall:    {R.mean():.4f}")
print(f"F1:        {F1.mean():.4f}")    