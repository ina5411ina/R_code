NeuronLayer <- setRefClass("NeuronLayer",

	fields = list(
		inputNum = "numeric",
		neuronNum = "numeric",
		weights = "matrix",
		bias = "matrix",
		mini_batch = "numeric",
		outputs = "matrix",
		inputs = "matrix",
		delta = "matrix",
		dInput = "matrix",
		adam_f = "numeric",
		adam_s = "numeric"
	),

	methods = list(
		initFields = function(inputNum, neuronNum, mini_batch){
			.self$inputNum <- inputNum
			.self$neuronNum <- neuronNum
			.self$mini_batch <- mini_batch
			.self$weights <- matrix(0.1, .self$inputNum, .self$neuronNum)
			.self$bias <- matrix(0.1, 1, .self$neuronNum)
			.self$outputs <- matrix()
			.self$inputs <- matrix()
			.self$delta <- matrix()
			.self$dInput <- matrix()
			.self$adam_s <- 0.0
			.self$adam_f <- 0.0
		},

		init_weight = function(){
			.self$adam_s <- 0.0
			.self$adam_f <- 0.0
			.self$weights <- matrix(runif(.self$inputNum*.self$neuronNum, 0.0, 1), .self$inputNum, .self$neuronNum)
			.self$bias <- matrix(runif(.self$neuronNum, 0.0, 1), 1, .self$neuronNum)
			# print(.self$weights)
		},

		ELU = function(x){
			for(i in 1:nrow(x)){
				for( j in 1:ncol(x)){
					if(x[i, j] < 0){
						x[i, j] = 1 * (exp(x[i,j]) -1)
					}
				}
			}

			return(x)
		},

		# # 計算每個layer的output
		feedForward = function(inputs){
			bb <- matrix(0, nrow(inputs), .self$neuronNum)
			for(i in 1:mini_batch){
				for(j in 1:.self$neuronNum){
					bb[i, j] <- .self$bias[1, j]
				}
			}
			# # .self$inputs neuron 進 activation function 前的值
			.self$inputs <- inputs%*%.self$weights + bb
			.self$outputs <- ELU(.self$inputs)
			return(.self$outputs)
		},

		calDelta = function(dLoss, dAct, dInput){
			# print("inputNum")
			# print(.self$inputNum)
			# print("neuronNum")
			# print(.self$neuronNum)
			# print(dLoss)
			# print(dAct)
			# print(dInput)
			a <- matrix(0, .self$inputNum, .self$neuronNum)
			b <- matrix(0, .self$inputNum, .self$neuronNum)
			c <- matrix(0, .self$inputNum, .self$neuronNum)
			for(i in 1:.self$inputNum){
				for(j in 1:.self$neuronNum){
					print(.self$neuronNum)
					a[i, j] <- dLoss[1, j]
					b[i, j] <- dAct[1, j]
				}
			}
			c[ ,1:.self$neuronNum] <- dInput

			# print(a)
			# print(b)
			# print(c)
			.self$dInput <- c
			.self$delta <- a * b
			# print(.self$delta)
			# print(.self$weights)
			# print(.self$bias)
		},

		update = function(LearningRate){
			gradient <-.self$delta * .self$dInput
			# print("gradient")
			# print(gradient)
			# # adam
			b1 <- 0.9
			b2 <- 0.99
			eps <- 1e-10
			totalG <- 0.0
			# for(i in 1:nrow(gradient)){
			# 	for(j in 1:ncol(gradient)){
			# 		# .self$adam_f <- b1 * .self$adam_f + (1-b1)*gradient[i,j]
			# 		# .self$adam_s <- b2 * .self$adam_s + (1-b2)*gradient[i,j]*gradient[i,j]
			# 		totalG <- totalG + gradient[i, j]
			# 	}
			# }
			totalG <- totalG + sum(gradient)
			totalG <- totalG/(nrow(gradient)*ncol(gradient))
			.self$adam_f <- b1*.self$adam_f + (1-b1)*totalG
			.self$adam_s <- b2*.self$adam_s + (1-b2)*totalG*totalG

			.self$weights <- .self$weights - (LearningRate * .self$adam_f/sqrt(.self$adam_s)+eps)
			.self$bias <- .self$bias - (LearningRate * .self$delta[1,])
			# print("wait update")
		}
	)
)

NeuralNetwork <- setRefClass("NeuralNetwork",

	fields = list(
		LearningRate = "numeric",
		neuronLayers = "vector"
	),

	methods = list(
		initFields = function(LearningRate){
			.self$LearningRate <- LearningRate
			.self$neuronLayers <- vector()
		},

		# # Layer 是一個物件 NeuronLayer
		addlayer = function(Layer){
			.self$neuronLayers <- append(.self$neuronLayers, Layer)
		},

		feedForward = function(inputs){
			# print("Foreard")
			for(i in .self$neuronLayers){
				inputs <- i$feedForward(inputs)
				# print(inputs)
			}
			return(inputs)
		},

		feedBackward = function(inputs, outputs){
			# print("Backward")
			lenNum <- length(.self$neuronLayers)
			mini <- .self$neuronLayers[[1]]$mini_batch
			preDaltas = vector()
			hidden <- 0
			while(lenNum){
				# print(lenNum)
				currentLayer <- .self$neuronLayers[lenNum]
				# # Wgradient = dLoss * dAct * dInput
				# # bias gradient = dLoss * Aact
				# # output layer
				# # ex: input: 3 output:4
				if(hidden == 0){
					# print("output layer")
					dLoss <- (outputs - inputs)/sqrt( (outputs - inputs)^2 )
				
					dLoss <- t (as.matrix( colSums(dLoss)/mini ) )  # # [1x4] 
					# print(dLoss)
					dAct <- t (as.matrix( colSums( dELU(inputs) )/ mini )) # # [1x4]
					# # next layer output
					dInput <- as.matrix( colSums( .self$neuronLayers[[lenNum - 1]]$outputs)/mini ) # # [3x1]

					cal <- currentLayer[[1]]$calDelta(dLoss, dAct, dInput)
					print(dLoss)

					cat("\n\n")
					

				} # # Hidden layer
				else{
					# print("Hidden layer")
					preLayer <- .self$neuronLayers[lenNum + 1]
					if(lenNum > 1){
						nextLayer <- .self$neuronLayers[lenNum - 1]
						nextoutput <- nextLayer[[1]]$outputs
					}
					else{
						nextoutput <- inputs
					}
					
					
					dLoss <- t (as.matrix( rowSums( preLayer[[1]]$delta * preLayer[[1]]$weights ) ) )# #[3x1]

					dAct <- t (as.matrix( colSums( dELU( currentLayer[[1]]$inputs ) )/mini))
					
					dInput <- as.matrix( colSums(nextoutput)/mini )
		
					currentLayer[[1]]$calDelta(dLoss, dAct, dInput)
				}
				hidden = 1
				lenNum <- lenNum - 1
			}
		},

		update = function(LearningRate){
			for(i in .self$neuronLayers){
				i$update(LearningRate)
			}
		},

		train = function(inputs){
			outputs <- feedForward(inputs)
			feedBackward(inputs, outputs)
			update(LearningRate)
		},

		RMSE = function(inputs){
			totalerror = 0
			outputs = feedForward(inputs)

			R = sqrt( (outputs - inputs)^2 )


			for(i in 1:nrow(inputs)){
				for( j in 1:ncol(inputs)){
					totalerror <- totalerror + R[i, j]
				}
			}
			# print(R/nrow(inputs)*ncol(inputs))
			return(totalerror/(nrow(inputs)*ncol(inputs)))
		},

		dELU = function(inputs){
			for(i in 1:nrow(inputs)){
				for(j in 1:ncol(inputs)){
					if(inputs[i, j] > 0){
						inputs[i, j] <- 1
					}
					else{
						inputs[i,j] <- exp(inputs[i,j])
					}
				}
			}
			return(inputs)
		},

		test_f = function(inputs){
			k <- 0

			for(i in .self$neuronLayers){
				inputs <- i$feedForward(inputs)
				k <- k+1
				if(k == length(.self$neuronLayers)/2){
					return(inputs)
				}
			}
		}	
	)
)

