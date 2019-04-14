import json
import random
import matplotlib.pyplot as plt

class Multiclass:
	def __init__(self, file, type_of_set):
		count, self.count_without_caps, self.different_labels = different_words(file, "train")
		self.number_of_perceptron = len(self.different_labels)
		self.perceptrons = {}
		for label in self.different_labels:
			self.perceptrons[label] = Perceptron(file, type_of_set, label, self.count_without_caps)
		for label in self.different_labels:
			self.perceptrons[label].train()

		self.ambiguous_words = ambiguous_words(file, "train")

	def predict(self, features):
		values = []
		for lab in self.different_labels:
			values.append(self.perceptrons[lab].w_dot_x(features))
		#print("enviado: ",self.different_labels[values.index(max(values))], str(max(values)))
		return self.different_labels[values.index(max(values))]

	#def test(self, file):
	def test(self, test_set):
		#test_set, idwd = get_set(file, "test")
		error = 0
		for elem in test_set:
			prediction = self.predict(self.perceptrons[self.different_labels[0]].get_feature_sparse_array(elem))
			if prediction != elem[-1]:
				#print("Mala prediccion: ", prediction, elem[-1], elem[0])
				error += 1
			# else:
			# 	#print("#BUENA")
		# print("Total error: ", error)
		# print("Test size: ", len(test_set))
		# print("Precision: ", (len(test_set)-error)*100.0/len(test_set))
		return len(test_set), error, (len(test_set)-error)*100.0/len(test_set) 

	def oov_measure(self, file):
		words, words_without_caps, different_labels = different_words(file, "test")
		oov_words = []
		for key in words_without_caps.keys():
			if key not in self.count_without_caps:
				if key not in oov_words:
					oov_words.append(key)
		test_set = get_set(file, "test")[0]
		test_set_oov = []
		for elem in test_set:
			if elem[0] in oov_words:
				if elem[0] not in test_set_oov:
					test_set_oov.append(elem)
		return self.test(test_set_oov) 

	def ambiguous_mesure(self, file):
		test_set = get_set(file, "test")[0]
		test_set_ambiguous = []
		for elem in test_set:
			if elem[0] in self.ambiguous_words:
				if elem[0] not in test_set_ambiguous:
					test_set_ambiguous.append(elem)
		return self.test(test_set_ambiguous) 

class Perceptron:
	def __init__(self, file, type_of_set, label_selected, count_without_caps):
		#count, count_without_caps, different_labels = different_words(file, "train")

		self.number_of_words = len(count_without_caps)
		self.number_of_features = 1 + self.number_of_words + (self.number_of_words + 2)*4 + len(suffix) + 0
		self.w = [0 for x in range(self.number_of_features)] #init weights
		self.a = 0
		self.max_epoch = 10
		self.training_set, self.id_word = get_set(file, "train")
		self.dev_set, idewf = get_set(file, "dev")
		self.label_selected = label_selected



		self.loss_function_evolve_training = []
		self.loss_function_evolve_validation = []
		self.w_memorie = []
		self.index_selected = 0

	def get_feature_sparse_array(self,elem_of_training_set):
		feature = [0] #bias element
		#print(elem_of_training_set[0])
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
		# log
		pos_used_until_know = 1 + self.number_of_words + (self.number_of_words + 2)*4 + len(suffix)


		return feature

	def train(self):
		for i in range(self.max_epoch):
			#print("Iteracion: ", i, "Perceptron: ", self.label_selected)
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
			self.w_memorie.append(self.w)

		# print("Evolucion del error en iteraciones: ", self.max_epoch)
		# print(self.loss_function_evolve_training)
		# print(self.w_memorie)

		#Putting as weight that with the lowet validation error
		self.w = self.w_memorie[self.loss_function_evolve_validation.index(min(self.loss_function_evolve_validation))]
		self.index_selected = self.loss_function_evolve_validation.index(min(self.loss_function_evolve_validation))
		#print("Index toamdo: ", self.loss_function_evolve_validation.index(min(self.loss_function_evolve_validation)))


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

#return different words by label
def get_by_label(label, sett):
	result = []
	for element in sett:
		if element[-1] == label:
			if element[0].lower() not in (result):
				result.append(element[0].lower())

	return result

#return dic with words and number of time found, also array with different labales
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

#return array with ambiguous words
def ambiguous_words(file, type_of_set):
	words_detected = {}
	set_to_count = json.load(open("./corpus/en/en."+file+"."+type_of_set+".json"))
	ambiguous_words = []
	for words, labels in set_to_count:
		for i in range(len(words)):
			if words[i].lower() in words_detected:
				if labels[i] not in words_detected[words[i].lower()]:
					words_detected[words[i].lower()].append(labels[i])
					if words[i].lower() not in ambiguous_words:
						ambiguous_words.append(words[i].lower())
			else:
				words_detected[words[i].lower()] = [labels[i]]

	return ambiguous_words

#put initial format to dataset 
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

def get_neighbout_pairs(file, type_of_set):
	s = json.load(open("./corpus/en/en."+file+"."+type_of_set+".json"))
	pairs_right = {}
	pairs_left = {}
	for words, labels in s:
		for i in range(len(words)):

			j = i-1
			aux = (words[i].lower()) +","+ (words[j].lower() if j >= 0 else "left_border")
			if aux in pairs_left:
				pairs_left[aux] +=1
			else:
				pairs_left[aux] = 1

			j = i+1
			aux = (words[i].lower()) + "," + (words[j].lower() if j < len(words) else "right_border") 
			if aux in pairs_right:
				pairs_right[aux] +=1
			else:
				pairs_right[aux] = 1
	return pairs_left, pairs_right


class Noisiness:
	def __init__(self,file_in_domain, file_out_domain):
		count_without_caps = different_words(file_in_domain,'train')[1]
		self.in_domain = count_without_caps
		count_without_caps = different_words(file_out_domain,'test')[1]
		self.out_domain = count_without_caps


	def out_of_Vocabulary(self):
		difference = 0
		for k in self.out_domain.keys():
			if k not in self.in_domain:
				difference+=1
		return difference



global suffix
suffix = ['acy', 'al', 'ance', 'ence', 'dom', 'er', 'or', 'ism', 'ist', 'ity', 'ty', 'ment', 'ness', 'ship', 'sion', 'tion', 'ate', 'en', 'ify', 'fy', 'ize', 'ise', 'able', 'ible', 'al', 'esque', 'ful', 'ic', 'ical', 'ious', 'ous', 'ish', 'ive', 'less', 'y']
#https://www.thoughtco.com/common-suffixes-in-english-1692725

def lecture(file, type_of_set):
	s = json.load(open("./corpus/en/en."+file+"."+type_of_set+".json"))
	for words, labels in s:
		for word in words:
			print(word, end=" ")
		print("")


if __name__ == "__main__":
	files_3_dataset = ["ewt", "gum", "lines", "partut"]
	files_only_test = ["foot", "pud"]
	#files_only_test = ["foot", "natdis", "pud"]
	type_of_set = ["train", "dev", "test"]

	i = 1
	multiclass = Multiclass(files_3_dataset[i], 3)
	#iteration selected
	iteration_selected = {}
	dev_error = {}
	train_error = {}
	# number_of_features = {}
	# number_of_words = {}
	for label in multiclass.perceptrons.keys():
		iteration_selected[label] = multiclass.perceptrons[label].index_selected
		dev_error[label] = multiclass.perceptrons[label].loss_function_evolve_validation
		train_error[label] = multiclass.perceptrons[label].loss_function_evolve_training
		# number_of_features[label] = multiclass.perceptrons[label].number_of_features
		# number_of_words[label] = multiclass.perceptrons[label].number_of_words

	print(files_3_dataset[i])
	print("File, number_test, fails_test, perc_test, number_ambiguous, fails_ambiguous, perc_ambiguous, number_oov, fails_oov, perc_oov")
	for file in (files_3_dataset+files_only_test):
		number_test, fails_test, perc_test = multiclass.test(get_set(file, "test")[0])
		number_ambiguous, fails_ambiguous, perc_ambiguous = multiclass.ambiguous_mesure(file)
		number_oov, fails_oov, perc_oov = multiclass.oov_measure(file)
		print(file +", "+ str(number_test)+", "+ str(fails_test)+", "+ str(perc_test)+", "+ str(number_ambiguous)+", "+ str(fails_ambiguous)+", "+ str(perc_ambiguous)+", "+ str(number_oov)+", "+ str(fails_oov)+", "+ str(perc_oov))

	print(", , , , , , , , , ,")
	print("Label, iteration_selected, train_error, dev_error")
	for label in multiclass.perceptrons.keys():
		print(label + ", " + str(iteration_selected[label]) + ", " + str(dev_error[label][iteration_selected[label]]) + ", " + str(train_error[label][iteration_selected[label]]))



		






	#different_words(files_3_dataset[0], type_of_set[0])
			
	# test, id_word = get_set(files_3_dataset[0], type_of_set[0])
	#print(test)
	#print(get_by_label("NOUN", test))
	# for elem in test:
	# 	print(elem)

	# p = Perceptron(files_3_dataset[0], 3, "NOUN")
	# p.train()
	# p.plot_training_and_validation_error()


	# multiclass = Multiclass(files_3_dataset[2], 3)
	# print("Precision over the whole set test: ")
	# multiclass.test(get_set(files_3_dataset[2], "test")[0])
	# print("Precision over OOV: ")
	# multiclass.oov_measure(files_3_dataset[2])
	# print("Precision over ambiguous words: ")
	# multiclass.ambiguous_mesure(files_3_dataset[2])

	# print(get_neighbout_pairs(files_3_dataset[0], "train"))
	# lecture(files_only_test[2], "test")
	# for i in range(5000):
	# 	print("I gotta stuck by loading datasets")



	


