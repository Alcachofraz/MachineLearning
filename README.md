# Machine Learning

## Introduction

Este projeto almeja explorar técnicas de aprendizagem automática para resolver determinados problemas. As técnicas estudadas e implementadas neste projeto residem no âmbito das redes neurais, aprendizagem por reforço, algoritmos genéticos e raciocínio automático para otimização e para planeamento.
Conforme as diferentes técnicas de aprendizagem automática vão sendo abordadas, determinados problemas vão sendo introduzidos e implementados. Para maior detalhe, consultar relatório em anexo.

## XOR (Redes Neurais)

Para implementar este problema, e qualquer outro problema deste projeto que envolva redes neurais, utilizar-se-á a biblioteca **SciKit-Learn** para a linguagem Python. Observe-se a figura que ilustra o programa implementado. **X** é um array com o conjunto de treino que consiste em **[[-1, -1], [-1, 1], [1, -1], [1, 1]]** e **Y** é um array com o output respetivo **[-1, 1, 1, -1]** (target data).  

![image](https://user-images.githubusercontent.com/75852333/154865458-dd0ba9c0-8fc8-433f-a643-8233e2f97b9e.png)
  
A rede neural contará com duas camadas escondidas, uma com 4 neurónios e outra com 2 neurónios. Mas porque não apenas uma camada com um neurónio? E porque não centenas de camadas com centenas de neurónios?  
  
Ora, seria incorreto pensar que quantas mais camadas e neurónios, melhor, dado que existe o fenómeno de **overfitting**. Este fenómeno ocorre quando a rede neural tenta aprender demasiados detalhes, incluindo o ruído, causando um desempenho muito pobre. No caso de utilizar apenas uma camada com um neurónio, correr-se-ia o risco de a rede neural não aprender detalhes suficientes, resultanto também num desempenho pobre. Este fenómeno designa-se **underfitting**.  
  
Utilizar-se-á um optimizador **SGD** (Stochastic Gradient Descent) que atualiza os pesos com recurso ao algoritmo de **back-propagation**.

A classe **MLPRegressor** aceita parâmetros como o momento, a taxa de aprendizagem, a apresentação aleatória, a função de activação e o algoritmo e resolução. O primeiro passo é estabelecer parâmetros padrão, para que se possa testá-los de forma individual e sucessiva. Os parâmetros padrão são:  
  
• Taxa de aprendizagem = **0.05**;  
• Momento = **0**;  
• Aleatoriedade = **False**;  
• Função de activação = **“tanh”** (visto que os dados estão em codificação bipolar **[-1 ,1]**);  
• Algoritmo de resolução = **“sgd”**;  
  
Dados estes parâmetros, e variando a taxa de aprendizagem obtêm-se resultados interessantes. Observando os gráficos da página seguinte, que representam a curva do erro associada à aprendizagem, conclui-se que a taxa de erro de **0.1** é atingida mais rapidamente, à medida que a taxa de aprendizagem aumenta. Isto acontece porque o modelo não fica preso em mínimos locais (pelo menos com tanta frequência) e consegue “saltar“ e ultrapassá-los, pois pratica uma aprendizagem mais intensa.  
  
Taxa de Aprendizagem = 0.05             |  Taxa de Aprendizagem = 0.25
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/75852333/154865171-f839088e-f6b3-43ed-905b-df9dd3b57f1b.png)  |  ![](https://user-images.githubusercontent.com/75852333/154865474-30f7d7ac-88cf-4931-a83a-9cc94f8e826e.png)
Taxa de Aprendizagem = 0.5             |  Taxa de Aprendizagem = 1.0
![](https://user-images.githubusercontent.com/75852333/154865480-d11a048e-969b-4856-b270-3e64875811e3.png)  |  ![](https://user-images.githubusercontent.com/75852333/154865483-e4860528-810f-4a0c-a97d-150d16bfe954.png)

Contudo, ao aumentar a taxa de aprendizagem em demasia, esses saltos tornam-se imprevisíveis, no sentido em que a aprendizagem torna-se mais desorganizada e menos metódica. Para a a taxa de aprendizagem de **1**, essa desorganização e imprevisibilidade faz-se notar. Por volta da iteração **32**, é descoberto um mínimo local, do qual o modelo sai imediatamente de forma desamparada (a taxa de erro sobe imenso porque a saída de um mínimo local implica o aumento da mesma), acabando por convergir num mínimo menor, na iteração **52**.  
  
No caso de uma taxa de aprendizagem igual a **2**, a representação gráfica é desnecessária pois o modelo torna-se inutilizável. Os “saltos” passam a ser tão grandes, e a aprendizagem tão desorganizada e imprevisível, que a descoberta de mínimos com uma taxa de erro abaixo de **0.1** é muito pouco provável.  
  
Em suma, conclui-se que o aumento da taxa de aprendizagem diminui o número de iterações necessárias para atingir a mesma taxa de erro, resultando num tempo de aprendizagem mais reduzido. Contudo, ao aumentar esse parâmetro em demasia, a aprendizagem e a procura de mínimos torna-se incerta e instável, resultando numa taxa de erro de convergência mais elevada.  
  
## Padrões (Redes Neurais)

Este problema consiste em ensinar a rede neural a reconhecer os padrões da figura e, se não se tratar de nenhum deles, deverá reconhecê-lo também. Ou seja, a rede neural receberá como dados de entrada um array bidimensional com quatro posições em ambas as direções (um quadrado, portanto), como mostra a figura. Na camada de saída haverão dois neurónios, um para cada padrão. Se a rede neural não detetar nenhum dos padrões deverá manter os neurónios inativos.

![image](https://user-images.githubusercontent.com/75852333/154865850-5f028783-ab66-4d7d-ae32-1ab603df9b60.png)

Como neste exercício, o conjunto de domínio é bastante mais vasto (existem múltiplas combinações para formar novos padrões), será necessário, primeiro que tudo, gerar uma função capaz de gerar padrões aleatórios. O conjunto de treino **X** será composto por, aproximadamente, por três quartos de padrões aleatórios, e um quarto de padrões predefinidos (**A** e **B**, em iguais quantidades), Quanto ao conjunto de dados de saída (target data), será composto por **[1, 0]** e **[0, 1]**, no caso dos padrões **A** e **B**, respetivamente, e por **[0, 0]**, no caso de um padrão diferente.

## Acordes Musicais (Redes Neurais)

Este problema procura reconhecer acordes tocados no piano (a informação das notas tocadas passa para o computador por **MIDI** e é interpretada pela biblioteca **Pygame** em Python), ou no teclado do computador (as teclas pressionadas são também reconhecidas pela mesma biblioteca).  
  
Na música (teoria musical), os acordes fundamentais são compostos por 3 notas. A tónica, a terceira e a quinta. Naturalmente, existem 12 notas diferentes (Dó, Dó Sustenido, Ré, Ré Sustenido, Mi, Fá, Fá Sustenido, Sol, Sol Sustenido, Lá, Lá Sustenido e Si), pelo que seria de esperar que existiriam 12 acordes diferentes (sendo cada um destes acordes a tónica de cada acorde). E é verdade. Contudo, existem acordes maiores e menores. Os acordes menores são semelhantes aos menores, com a excepção de uma das notas. Tome-se como exemplo o acorde de dó maior (C MAJOR) e o acorde de dó menor (C MINOR). A tónica e a quinta mantêm-se. Apenas a terceira muda, sendo que no acorde menor desce meio tom (uma tecla).  

Acordes             |  Tradução
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/75852333/154865924-91220821-0f7c-4c1e-9b33-5401c57efea2.png)  |  ![](https://user-images.githubusercontent.com/75852333/154865950-462f3949-9d41-4fcd-b543-d47fa06702ca.png)

Note-se a equivalência da terminologia portuguesa e inglesa representada na tabela , referente à denominação dos acordes. Com toda esta informação, é possível concluir o seguinte:  
  
• Em qualquer acorde, a quinta dista 7 meio-tons da tónica;  
• Nos acordes maiores, a terceira dista 4 meio-tons da tónica;  
• Nos acordes menores, a terceira dista 3 meio-tons da tónica;  
• Existem 24 acordes diferentes (maiores e menores)Acrescenta-se também que uma oitava é a distância entre duas notas iguais, isto é, de Dó a Dó, por exemplo. Por isso, uma oitava conta com 12 notas diferentes. Isto significa que o espaço de uma oitava é suficiente para representar todos os 24 acordes.  

![image](https://user-images.githubusercontent.com/75852333/154866017-f7ae7eb0-85da-48a8-a7fc-0e0b26e401d0.png)

Uma oitava será representada pela seguinte lista: **[*, *, *, *, *, *, *, *, *, *, *, *]** com dimensão **12**. Os dados de entrada terão este formato, em que as posições ativadas (com valor **1**) representarão notas a soar, e as posições desativadas (com valor **0**) representarão notas em silêncio. Dado que haverá **24** acordes possíveis, a camada de saída contará com **24** neurónios, um para cada acorde. Os primeiros **12** serão maiores (de Dó a Si) e os segundos **12** serão menores (também de Dó a Si).  
  
O objetivo da rede neural a implementar é, com base na oitava de entrada, concluir acerca do acorde que está a ser tocado. Por exemplo, o acorde Dó Maior (que contém a tónica Dó, a terceira Mi, e a quinta Sol), se fosse tocado, a entrada da rede neural seria [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0] e a predição correta seria [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].  
  
Como já foi referido, para recolher informação acerca das teclas tocadas tanto no piano, como no teclado do computador, utilizar-se-á a biblioteca **Pygame** do Python.  
  
 ![image](https://user-images.githubusercontent.com/75852333/154866112-db137f36-8e7b-48ff-be64-27065ba84e79.png)

Será necessária uma função que, com base numa oitava, seja capaz de gerar um array com **24** posições indicando o acorde que está a ser tocado, para mais facilmente gerar o conjunto de dados **Y** (target data). Será também necessária uma função para gerar acordes aleatórios e uma função para, com base numa oitava, identificar o acorde que está a ser tocado (o nome do acorde).  
  
Será então desenvolvido um pequeno programa em Python para treinar o modelo e para ouvir eventos no piano (ou no teclado do computador) relativos a teclas premidas, mostrando sempre o acorde que está a ser tocado.  
  
Para o modelo da rede neural serão escolhidas duas camadas escondidas, uma com **128** neurónios e outra com **64** neurónios, será escolhida a função de ativação **relu** (codificação binária), uma taxa de aprendizagem de **0.05** e um momentum de **0.5**.

Ao executar o programa, o modelo é treinado e surge uma pequena interface gráfica que mostra a localização de cada nota no teclado do computador, bem como o acorde que está a ser tocado no momento. Na figura da esquerda, é mostrado **“UNKNOWN”**, visto que nenhuma tecla está a ser premida e nenhum acorde está a soar. Contudo, na figura do meio observa-se que, ao premir a tecla **Q**, a tecla **E** e a tecla **T**, a rede neural identifica o acorde **C MAJOR**.

Nenhum acorde a ser tocado   |  Dó maior  |  Si maior
:-------------------------:|:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/75852333/154866145-7ed7756e-3080-478b-a69c-a670f159e3ce.png)  |  ![](https://user-images.githubusercontent.com/75852333/154866234-964dc4b4-ee7e-4365-9fa6-045d899c638e.png)  |  ![](https://user-images.githubusercontent.com/75852333/154866238-89f582b6-b610-4ca6-9616-ba732e5407ad.png)


