source("Neuron.R")


TrainData <- read.csv("Train.csv", header = TRUE, sep = ",")

LRate <- 1e-5
mini_batch <- 128
Epoch <- 10
seed <- 1

Autoencoder <- NeuralNetwork$new(LearningRate = LRate)
set.seed(seed)

H1 <- NeuronLayer$new(inputNum = 32, neuronNum = 8, mini_batch = mini_batch)
H1$init_weight()
H2 <- NeuronLayer$new(inputNum = 8, neuronNum = 2, mini_batch = mini_batch)
H2$init_weight()
H3 <- NeuronLayer$new(inputNum = 2, neuronNum = 8, mini_batch = mini_batch)
H3$init_weight()
O4 <- NeuronLayer$new(inputNum = 8, neuronNum = 32, mini_batch = mini_batch)
O4$init_weight()

Autoencoder$addlayer(H1)
Autoencoder$addlayer(H2)
Autoencoder$addlayer(H3)
Autoencoder$addlayer(O4)

# # ---------------------------------------------------------------------------------------

print("Start Training------------------")
# runif(10, 0.0, 1)
cnt = 1
while(cnt <= Epoch){
	run <- (nrow(TrainData)/mini_batch)
	for(i in 1:run){
		a <- 1 + (mini_batch*(i-1))
		b <- mini_batch + (mini_batch*(i-1))
		inputs <- as.matrix(TrainData[a:b, ])
		Autoencoder$train(inputs)
		Loss <- Autoencoder$RMSE(inputs)
		# print(Loss)
		# cat("\n\n")
	}
	cat("Epoch : ", cnt, " ", Loss, "\n")
	cnt <- cnt + 1

}


# for(i in 1:2){
# 	a <- 1 + (mini_batch*(i-1))
# 	b <- mini_batch + (mini_batch*(i-1))
# 	inputs <- as.matrix(TrainData[a:b, ])
# 	if(i == 1){
# 		m1 <- Autoencoder$test_f(inputs)
# 		print(m1)
# 	}
# 	m1 <- rbind(m1, Autoencoder$test_f(inputs))
# }

# write.csv(m1, "train_output.csv", row.names=FALSE)


