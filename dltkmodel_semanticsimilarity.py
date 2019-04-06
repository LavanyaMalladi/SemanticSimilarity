import abc

class DITKModel_SemanticSimilarity(abc.ABC):

	# This class defines the common behavior that the sub-classes in the family of Semantic Similarity algorithms can implement/inherit

	"""
    BENCHMARKS:
        --------------------------------------------------------------------------------------------------------------------------------------------
        |   DATASET            |         FORMAT                    |               EXAMPLE                      |    EVALUATION METRICS           |
        |------------------------------------------------------------------------------------------------------------------------------------------- 
        | SemEval 2017 Task1   | Sentences and its similarity score |['Dogs are fighting','Dogs are wrestling',4]| Pearson Correlation Coefficient |           |
        |------------------------------------------------------------------------------------------------------------------------------------------
        | SemEval 2014	       | Sentences and its similarity score |['Dogs are fighting','Dogs are wrestling',4]| Pearson Correlation             | 
        |                                                                                                                Coefficient                |
        |-------------------------------------------------------------------------------------------------------------------------------------------
        | SICK Dataset         | Sentences and its similarity score |['Dogs are fighting','Dogs are wrestling',4]| Pearson Correlation             |
        |                                                                                                               Coefficient                |
        --------------------------------------------------------------------------------------------------------------------------------------------
    	"""

    	"""

    GENERAL FUNCTIONAL FLOW :

	Read dataset --> Train Model(if required) --> Predict Similarity Values --> Evaluate Pearson Correlation Coefficient -->
	--> Display Evaluation Score


    	"""

	@abc.abstractmethod
	def read_dataset(self, fileNames, *args, **kwargs): 
		"""
		Reads a dataset that is a CSV/Excel File.

		Args:
			fileName : With it's absolute path

		Returns:
			training_data_list : List of Lists that containes 2 sentences and it's similarity score 
			Note :
				Format of the output : [[S1,S2,Sim_score],[T1,T2,Sim_score]....]

		Raises:
			None
		"""
		#parse files to obtain the output
		return training_data_list

	@abc.abstractmethod 	
	def train(self,*args, **kwargs):  #<--- implemented PER class
	
	    # some individuals don't need training so when the method is extended, it can be passed
		
        	pass

	@abc.abstractmethod
	def predict(self, data_X, data_Y, *args, **kwargs):  
		"""
		Predicts the similarity score on the given input data(2 sentences). Assumes model has been trained with train()

		Args:
			data_X: Sentence 1(Non Tokenized).
			data_Y: Sentence 2(Non Tokenized)

		Returns:
			prediction_score: Similarity Score ( Float ) 
				
		Raises:
			None
		"""
		return prediction_score

	@abc.abstractmethod
	def evaluate(self, actual_values, predicted_values, *args, **kwargs): 
		"""
		Returns the correlation score(0-1) between the actual and predicted similarity scores

		Args:
			actual_values : List of actual similarity scores
			predicted_values : List of predicted similarity scores

		Returns:
			correlation_coefficient : Value between 0-1 to show the correlation between the values(actual and predicted)

		Raises:
			None
		"""
		return evaluation_score
		

"""
# Sample workflow:

inputFiles = ['thisDir/file1.txt','thatDir/file2.txt','./file1.txt']

myModel =   DITKModel_SemanticSimilarity()# instatiate the class

train_data_list = myModel.read_dataset(inputFiles,'train')  # read in a dataset for training

test_data_list = myModel.read_dataset(inputFiles,'test')  # read in a dataset for testing

myModel.train(train_data_list)  # trains the model and stores model state in object properties or similar

predictions_score = myModel.predict(test_sentence1_list, text_sentence2_list)  # generate predictions! output format will be same for everyone

evaluation_score = myModel.evaluate(predictions_score, test_actual_score)  # calculate evaluation_score

print(evaluation_score)

"""

