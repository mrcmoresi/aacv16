# KMEANS

# ##C = 10
# clusters size =  [1, 5, 10, 25, 50, 75, 100, 110, 125, 140, 150, 200, 300]
# accuracy =  [0.0871, 0.3303, 0.4750, 0.5701, 0.6261, 0.6398, 0.6469, 0.6536, 0.6472, 0.6559, 0.6633, 0.6361, 0.6425]
# list elapsed time =  [4.47, 6.04, 7.8901, 13.40, 22.50, 31.51, 40.49, 44.37, 48.22, 52.72, 56.84, 73.86, 114.83]

import matplotlib.pyplot as plt

#C=1.5
clusters_size = [1,5,10,25,50,75,100,110,125,140,150,175,200,250,300,325]
accuracy = [0.0472,0.4177,0.4606,0.5504,0.6100,0.6428,0.6566,0.6753,0.6646,0.659,0.683,0.682,0.683,0.679,0.663,0.661]
elapsed_time = [4.4957,6.3762,8.1321,13.7002,22.8868,30.9782,40.1747,42.3768,48.1458,54.9081,57.8346,66.0863,75.4143,92.0769,109.527,118.4008]

# list_elapsed_time = []
# for i in range(len(elapsed_time)):
# 	list_elapsed_time.append(elapsed_time[i]/60)

#Imprimo tradeoff cluster accuracy como puntos
plt.plot(clusters_size,accuracy,'g^')
#Imprimo tradeoff cluster tiempo como triangulos
#plt.plot(clusters_size, list_elapsed_time, 'ro')

#Imprimo la funcion que se genera entre cluster y accuracy
plt.plot(clusters_size, accuracy)

#plt.plot(clusters_size, list_elapsed_time)

plt.xlabel('N_Clusters')
plt.ylabel('Accuracy ')

plt.grid()
plt.scatter(clusters_size,accuracy)
#plt.scatter(clusters_size,list_elapsed_time)

plt.show()
