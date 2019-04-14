import json
import random
import matplotlib.pyplot as plt

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

	# print("Number of differents words: ", len(count))
	# print("Number of differents lowercase words: ", len(count_without_caps))
	# print("Number of differents labels: ", len(different_labels))

	return count, count_without_caps, different_labels

def get_set(file, type_of_set):
	training_set = []			#array that maps a word with some characteristics
	s = json.load(open("./corpus/en/en."+file+"."+type_of_set+".json"))
	id_word_dict = {}
	ids = 1
	for words, labels in s:
		for i in range(len(words)):
			if words[i].lower() not in id_word_dict:
				id_word_dict[words[i].lower()] = ids
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

class Multiclass:
	def __init__(self, file, type_of_set):
		count, count_without_caps, self.different_labels = different_words(file, "train")
		self.number_of_perceptron = len(self.different_labels)
		count, count_without_caps, self.different_labels = different_words(file, "train")
		self.perceptrons = {}
		for label in self.different_labels:
			self.perceptrons[label] = Perceptron(file, type_of_set, label, count_without_caps)
		for label in self.different_labels:
			self.perceptrons[label].train()

		self.test_set, idewf = get_set(file, "test")

	def predict(self, features):
		values = []
		for lab in self.different_labels:
			values.append(self.perceptrons[lab].w_dot_x(features))
		#print("enviado: ",self.different_labels[values.index(max(values))], str(max(values)))
		return self.different_labels[values.index(max(values))]

	def test(self):
		error = 0
		for elem in self.test_set:
			prediction = self.predict(self.perceptrons[self.different_labels[0]].get_feature_sparse_array(elem))
			if prediction != elem[-1]:
				print("Mala prediccion: ", prediction, elem[-1], elem[0])
				error += 1
			# else:
			# 	#print("#BUENA")
		print("Total error: ", error)
		print("Test size: ", len(self.test_set))

	def get_feature_sparse_array(self,elem_of_training_set):
		feature = [0] #bias element
		if elem_of_training_set[0].lower() in self.id_word:
			feature.append(self.id_word[elem_of_training_set[0].lower()])

		for i in range(4):
			if elem_of_training_set[1][i].lower() in self.id_word:
				feature.append(self.number_of_words*(1+i) + (2*i) + self.id_word[elem_of_training_set[1][i].lower()])
			elif elem_of_training_set[1][i].lower() == 'left_border':
				feature.append(self.number_of_words*(2+i) + (2*i) + 1)
			elif elem_of_training_set[1][i].lower() == 'right_border':
				feature.append(self.number_of_words*(2+i) + (2*i) + 2)



class Perceptron:
	def __init__(self, file, type_of_set, label_selected, count_without_caps):
		#count, count_without_caps, different_labels = different_words(file, "train")

		self.number_of_words = len(count_without_caps)
		self.number_of_features = 1 + self.number_of_words + (self.number_of_words + 2)*4 + len(suffix) + 0
		self.w = [0 for x in range(self.number_of_features)] #init weights
		self.a = 0
		self.max_epoch = 5
		self.training_set, self.id_word = get_set(file, "train")
		self.dev_set, idewf = get_set(file, "dev")
		self.label_selected = label_selected



		self.loss_function_evolve_training = []
		self.loss_function_evolve_validation = []

	def get_feature_sparse_array(self,elem_of_training_set):
		feature = [0] #bias element
		if elem_of_training_set[0].lower() in self.id_word:
			feature.append(self.id_word[elem_of_training_set[0].lower()])

		for i in range(4):
			if elem_of_training_set[1][i].lower() in self.id_word:
				feature.append(self.number_of_words*(1+i) + (2*i) + self.id_word[elem_of_training_set[1][i].lower()])
			elif elem_of_training_set[1][i].lower() == 'left_border':
				feature.append(self.number_of_words*(2+i) + (2*i) + 1)
			elif elem_of_training_set[1][i].lower() == 'right_border':
				feature.append(self.number_of_words*(2+i) + (2*i) + 2)

		# suffix
		pos_used_until_know = 1 + self.number_of_words + (self.number_of_words + 2)*4
		for i in range(len(suffix)):
			if elem_of_training_set[0].lower().endswith(suffix[i]):
				#print(elem_of_training_set[0].lower())
				feature.append(pos_used_until_know + i)


		# other

		return feature

	def train(self):
		for i in range(self.max_epoch):
			print("Iteracion: ", i, "Perceptron: ", self.label_selected)
			random.shuffle(self.training_set)
			for elem in self.training_set:
				x = self.get_feature_sparse_array(elem)
				y = 1 if self.w_dot_x(x) >=0 else -1
				#print(self.w_dot_x(x));
				y_i = 1 if elem[-1] == self.label_selected else -1
				if y != y_i:
					self.modify_weight(y_i, x)

			self.loss_function_evolve_training.append(self.loss_error(self.training_set))
			self.loss_function_evolve_validation.append(self.loss_error(self.dev_set))
		# print("Evolucion del error en iteraciones: ", self.max_epoch)
		# print(self.loss_function_evolve_training)

	def w_dot_x(self, features):
		summ = 0
		for i in features:
			summ += self.w[i]

		return summ

	def modify_weight(self, y_i, x):
		for number in x:
			self.w[number] += y_i #because we are working with binary values

	def loss_error(self, sett):
		error = 0
		for elem in sett:
			x = self.get_feature_sparse_array(elem)
			y = 1 if self.w_dot_x(x) >=0 else -1
			y_i = 1 if elem[-1] == self.label_selected else -1
			# print(sett[-1], self.label_selected)
			# print(y,y_i)
			if y != y_i:
				error += 1
		#print("error", error)
		return error / len(sett)


	def plot_training_and_validation_error(self):
		data = {}
		data['dev'] = self.loss_function_evolve_validation
		data['train'] = self.loss_function_evolve_training
		data['x'] = range(self.max_epoch)
		plt.plot( 'x', 'dev', data=data, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
		plt.plot( 'x', 'train', data=data, marker='o', color='red', linewidth=2)
		plt.show()

def get_by_label(label, sett):
	result = []
	for element in sett:
		if element[-1] == label:
			if element[0].lower() not in (result):
				result.append(element[0].lower())

	return result



global suffix
suffix = ['acy', 'al', 'ance', 'ence', 'dom', 'er', 'or', 'ism', 'ist', 'ity', 'ty', 'ment', 'ness', 'ship', 'sion', 'tion', 'ate', 'en', 'ify', 'fy', 'ize', 'ise', 'able', 'ible', 'al', 'esque', 'ful', 'ic', 'ical', 'ious', 'ous', 'ish', 'ive', 'less', 'y']
#https://www.thoughtco.com/common-suffixes-in-english-1692725

if __name__ == "__main__":
	files_3_dataset = ["ewt", "gum", "lines", "partut"]
	files_only_test = ["foot", "natdis", "pud"]
	type_of_set = ["train", "dev", "test"]
	#different_words(files_3_dataset[0], type_of_set[0])

	# test, id_word = get_set(files_3_dataset[0], type_of_set[0])
	#print(test)
	#print(get_by_label("NOUN", test))
	# for elem in test:
	# 	print(elem)

	# p = Perceptron(files_3_dataset[0], 3, "NOUN")
	# p.train()
	# p.plot_training_and_validation_error()
	multiclass = Multiclass(files_3_dataset[0], 3)
	multiclass.test()


