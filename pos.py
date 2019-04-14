import json
import random

def different_words(file, type_of_set):
	set_to_count = json.load(open("./corpus/en/en."+file+"."+type_of_set+".json"))
	count = {}					#dict with differents words
	count_without_caps = {}		#dict with differents lowercase words
	different_labels = []		#array with labels

	for words, labels in set_to_count:
		for word in words:
			if word.lower() in count_without_caps:
				count_without_caps[word.lower()] += 1
			else:
				count_without_caps[word.lower()] = 1

			if word in count:
				count[word] += 1
			else:
				count[word] = 1

		for label in labels:
			if label not in different_labels:
				different_labels.append(label) 

	print("Number of differents words: ", len(count))
	print("Number of differents lowercase words: ", len(count_without_caps))
	print("Number of differents labels: ", len(different_labels))

	return count, count_without_caps, different_labels

def get_trainning_set(file, type_of_set):
	training_set = []			#array that maps a word with some characteristics
	s = json.load(open("./corpus/en/en."+file+"."+type_of_set+".json"))
	id_word_dict = {}
	ids = 1
	for words, labels in s:
		for i in range(len(words)):
			if words[i] not in id_word_dict:
				id_word_dict[words[i]] = ids
				ids += 1

			aux = [words[i]]
			enviroment = []
			for j in range(i - 2, i + 3):
				if j == i:
					continue
				if j < 0:
					enviroment.append("left_border")
				elif j >= len(words):
					enviroment.append("right_border")
				else:
					enviroment.append(words[j])
			aux.append(enviroment)
			aux.append(labels[i])
			training_set.append(aux)

	return training_set, id_word_dict

class Perceptron:
	def __init__(self, number_of_words, file, type_of_set, label_selected):
		self.number_of_words = number_of_words
		self.number_of_features = 1 + number_of_words + (number_of_words + 2)*4 + 0 + 0
		self.w = [0 for x in range(self.number_of_features)] #init weights
		self.a = 0
		self.max_epoch = 50
		self.training_set, self.id_word = get_trainning_set(file, type_of_set)
		self.label_selected = label_selected

		loss_function_evolve_training = []
		loss_function_evolve_validation = []

	def get_feature_sparse_array(self,elem_of_training_set):
		feature = [0] #bias element
		if elem_of_training_set[0] in self.id_word:
			feature.append(self.id_word(elem_of_training_set[0]))

		for i in range(4):
			if elem_of_training_set[1][i] in self.id_word:
				feature.append(self.number_of_words*(1+i) + (2*i) + self.id_word(elem_of_training_set[1][i]))
			elif elem_of_training_set[1][i] == 'left_border':
				feature.append(self.number_of_words*(2+i) + (2*i) + 1)
			elif elem_of_training_set[1][i] == 'right_border':
				feature.append(self.number_of_words*(2+i) + (2*i) + 2)

		# suffix
		# other

		return feature

	def training(self):
		for i in range(self.max_epoch):
			random.shuffle(self.training_set)
			for i in range(len(self.training_set)):
				x = self.get_feature_sparse_array(self.training_set[i])
				y = 1 if self.w_dot_x(x) >=0 else -1
				y_i = 1 if self.training_set[-1] == self.label_selected else -1
				if y != y_i:
					self.modify_weight(y_i, x)


	def w_dot_x(self, features):
		summ = 0
		for i in features:
			summ += self.w[i]

		return summ

	def modify_weight(self, y_i, x):
		for number in x:
			self.w[number] += y_i





	# def loss_error(self):


	# def train(self):

	# def dot_product(self, features):

def get_by_label(label, sett):
	result = []
	for element in sett:
		if element[-1] == label:
			result.append(element[0])

	return result




if __name__ == "__main__":
	files_3_dataset = ["ewt", "gum", "lines", "partut"]
	files_only_test = ["foot", "natdis", "pud"]
	type_of_set = ["train", "dev", "test"]
	#different_words(files_3_dataset[0], type_of_set[0])

	test, id_word = get_trainning_set(files_3_dataset[0], type_of_set[0])
	#print(test)
	print(get_by_label("NOUN", test))
	# for elem in test:
	# 	print(elem)


