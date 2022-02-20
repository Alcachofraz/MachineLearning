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

A classe MLPRegressor aceita parâmetros como o momento, a taxa de aprendizagem, a apresentação aleatória, a função de activação e o algoritmo e resolução. O primeiro passo é estabelecer parâmetros padrão, para que se possa testá-los de forma individual e sucessiva. Os parâmetros padrão são:  
  
• Taxa de aprendizagem = 0.05;  
• Momento = 0;  
• Aleatoriedade = False;  
• Função de activação = “tanh” (visto que os dados estão em codificação bipolar [-1 ,1]);  
• Algoritmo de resolução = “sgd”;  
  
Dados estes parâmetros, e variando a taxa de aprendizagem obtêm-se resultados interessantes. Observando os gráficos da página seguinte, que representam a curva do erro associada à aprendizagem, conclui-se que a taxa de erro de 0.1 é atingida mais rapidamente, à medida que a taxa de aprendizagem aumenta. Isto acontece porque o modelo não fica preso em mínimos locais (pelo menos com tanta frequência) e consegue “saltar“ e ultrapassá-los, pois pratica uma aprendizagem mais intensa.  
  
<div class="row">
  <div class="column">
    <img src="https://user-images.githubusercontent.com/75852333/154865171-f839088e-f6b3-43ed-905b-df9dd3b57f1b.png" alt="0.05" style="width:50%">
  </div>
  <div class="column">
    <img src="https://user-images.githubusercontent.com/75852333/154865474-30f7d7ac-88cf-4931-a83a-9cc94f8e826e.png" alt="0.25" style="width:50%">
  </div>
</div>

![image](https://user-images.githubusercontent.com/75852333/154865480-d11a048e-969b-4856-b270-3e64875811e3.png)

![image](https://user-images.githubusercontent.com/75852333/154865483-e4860528-810f-4a0c-a97d-150d16bfe954.png)




