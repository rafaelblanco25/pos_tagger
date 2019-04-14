import json

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


class Perceptron:
	def __init__(self, number_of_labels):
		w = [0 for x in range(number_of_labels)] #init weights
		loss_function_evolve = []

	def train(self):

	def dot_product(self, features):


if __name__ == "__main__":
	files_3_dataset = ["ewt", "gum", "lines", "partut"]
	files_only_test = ["foot", "natdis", "pud"]
	type_of_set = ["train", "dev", "test"]
	different_words(files_3_dataset[0], type_of_set[0])
