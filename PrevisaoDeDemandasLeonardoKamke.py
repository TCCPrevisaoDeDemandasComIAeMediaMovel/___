
import csv

#import numpy as np
#import matplotlib.pyplot as plt
#from numpy.core.records import array
from sklearn import metrics
#from sklearn.metrics import r2_score                            #https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
from sklearn.neural_network import MLPClassifier                #https://github.com/jiuxianghedonglu/MLPClassifier.python
from sklearn.neural_network import MLPRegressor                 #https://analyticsindiamag.com/a-beginners-guide-to-scikit-learns-mlpclassifier/
#from sklearn.model_selection import train_test_split

"""
Documentação do MLPRegressor, para ajustar os parametros, melhorar assertividade.
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html

"""
clf2 = MLPRegressor(solver='lbfgs',alpha=1e-5, random_state=1, activation='tanh',hidden_layer_sizes=(100,), learning_rate = 'adaptive')
clf1 = MLPRegressor(hidden_layer_sizes=(5,),
                                       activation='relu',
                                       solver='adam',
                                       learning_rate='adaptive',
                                       max_iter=1000,
                                       learning_rate_init=0.01,
                                       alpha=0.01)

clf = MLPRegressor(
    hidden_layer_sizes=(100,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
    random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#array = np.array #usado no sellprediction
dados = {} #dicionario de dados
with open('C:\Temp\TesteLeonardo-edit-3COPIA.csv') as csv_file: #le o arquivo no diretorio tal...
    csv_reader = csv.reader(csv_file, delimiter=',') #le o arquivo csv separado por virgula
    for row in csv_reader:
        if row[0] not in dados: #se nao existe esse campo no dicionario, criar
                dados[row[0]]=[] #cria a entrada no dicionario
        for column in row[1:]: #pula a primeira coluna
            if row[0] != 'periodo': #pega a primeira coluna pulando o "periodo que é a coluna 0"
                  dados[row[0]].append(float(column)) #preenchendo com os valores de cada coluna de acordo com a linha
            else:
                dados[row[0]].append(column) #preenchendo com os valores de cada coluna de acordo com a linha, se for diferente de periodo. 
#Documentação LIDA -> https://dadosedecisoes.com.br/manipulacao-de-arquivos-csv-em-python/
print("Produtos disponiveis para previsão:",'p116300','p162660','p271706','p449606','p1021500','p25130600','p25140600') #imprime as opções para prever 
campo = input("Qual produto voce quer treinar/prever? ")  #Após o campo inserido, será usado no for, para junto com as variáveis(Inflacao, INCC e etc) serem treinados...
print("--- TREINANDO --- ") #LER https://analyticsindiamag.com/a-beginners-guide-to-scikit-learns-mlpclassifier/
print("['Periodo','inflção','INCC','Dólar','IPCA','ICF','ICC','CVCS']= Dados Vendas/Treino.\t")
metricsNN = [] # metrics for Neural Networks ['inflacao','INCC','dolar','IPCA','ICF','ICC','CVCS']
sellPrediction = [] # sell values for training
for pos in range(len(dados[campo])): #documentação https://pynative.com/python-range-function/
    print("\t%f,\t%f,\t%f,[\t%fp]" %(dados['dolar'][pos],dados['inflacao'][pos],dados['IPCA'][pos],dados['CVCS'][pos]))
    metricsNN.append([dados['dolar'][pos],dados['inflacao'][pos],dados['IPCA'][pos],dados['CVCS'][pos]])
#Exemplos em: https://www.programcreek.com/python/example/93778/sklearn.neural_network.MLPRegressor
tam_treino = int (len(metricsNN)*70/100) #70%  
print("----------------------------------------------")
print("ENTRADAS:",metricsNN[:tam_treino])
print("----------------------------------------------")
print("VENDAS:",dados[campo][:tam_treino])
x_train = metricsNN[:tam_treino]
y_train = dados[campo][:tam_treino]
clf.fit(x_train,y_train) #treinando com 70% dos dados
print("----------------------------------------------")
print("Tamanho Treino: ",len(metricsNN[:tam_treino]))
print("----------------------------------------------")
print("Tamanho Teste:  ",len(metricsNN[tam_treino:]))
print("----------------------------------------------")
for teste in metricsNN[tam_treino:]:
     sellPrediction.append(clf.predict([teste])) #predições com os outros 30%   
sellPredictiontransformado = [sellPrediction[i][0] for i in range(len(sellPrediction))] #transformo o array em valores normais
sellPredictionDeterminado = list(map(lambda x: x * 1000, sellPredictiontransformado)) #trata os dados 
sellPredictionDet = [round(num, 1) for num in sellPredictionDeterminado] #arredondamento dos elementos da lista
real = dados[campo][tam_treino:]
print("Real: ",real)
print("----------------------------------------------")
print("Sell:",sellPredictionDet)
#print("Previsao: ",(sellPrediction,r2_score))
print("----------------------------------------------")
#coefficient_of_dermination = r2_score(real, sellPredictionDet)
#print("R-squared: ",coefficient_of_dermination)
#coefficient_of_dermination = np.corrcoef(real, sellPredictionDet)
#print("R-squared: ", coefficient_of_dermination)
print("----------------------------------------------")
"""
fig = plt.figure()
axes = plt.gca()
ax1 = fig.add_subplot(111)
ax1.scatter(real,sellPredictionDet, s=10, c='r', marker="o", label='Previsao')
plt.scatter(sellPredictionDet, real, color='black' )
#l, color='blue', linewidth=3)
#plt.bar(sellPredictionDet,real, label = 'Previsão', color = 'r')
x= sellPredictionDet
y= real
bar_real = plt.bar(real,real ,width=1000, label = 'Real', color = 'red')
bar_sell = plt.bar(sellPredictionDet,sellPredictionDet, width=1000, label = 'Previsão', color = 'blue')
labels = ['Real', 'vendido']
labels = labels
location = x+y
plt.xticks(location, labels)
plt.legend()
plt.show()"""
