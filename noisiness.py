
import json
import random
import math 

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



class Noisiness:	

	def out_of_Vocabulary(self):
		difference = 0
		for k in self.out_domain.keys():
			if k not in self.in_domain:
				difference+=1
		return difference

	def generate_n_grams_char(self,word,n): #Used for the KL divergence
		grams=[] #array with the all the possible n-grams in word
		grams= [word[i:i+n] for i in range(len(word)-n+1)] 
		return grams

	def grams_3_in_dicts(self, train_data, test_data): #Used for the KL divergence
		all_grams={} #dict with all the 3-grams in a dictionary
		for key in train_data.keys():
			self.array_grams=self.generate_n_grams_char(key,3) #array aux to recieve the grams of an individual word in the dict
			for i in range(len(self.array_grams)):
				if self.array_grams[i] in all_grams:
					all_grams[self.array_grams[i]] += train_data[key]
				else:
					all_grams[self.array_grams[i]] = train_data[key]

		for key in test_data.keys():
			self.array_grams=self.generate_n_grams_char(key,3) #array aux to recieve the grams of an individual word in the dict
			for i in range(len(self.array_grams)):
				if self.array_grams[i] in all_grams:
					all_grams[self.array_grams[i]] += test_data[key]
				else:
					all_grams[self.array_grams[i]] = test_data[key]

		N = sum(all_grams.values()) #sum of all the 3-gram of characters in the trainand test sets
		V = len (all_grams) #number of distinct 3-grams of characters in the train and in the test sets
		
		return N, V, all_grams

	def number_chars_in_set(self,dictionary): #Used for the KL divergence
		chars_number=0
		for key in dictionary.keys():
			for l in key:
				chars_number += dictionary[key]
		return chars_number

	def prob_set(self, N,V,numberOf_key,dataset): #the probability to observe the 3-gram (ci−2ci−1ci) in data set dwith Laplace-smoothing;
		d = self.number_chars_in_set(dataset)
		
		return (numberOf_key+1)/(N+V*(d-2))

	def KL_divergence(self, c_test, c_train): #KL divergence of two sets
		all_grams = {}
		# prob_test = []
		# prob_train = []
		KL_array = []

		N, V, all_grams = self.grams_3_in_dicts(c_test,c_train) 

		count = 0
		for key in all_grams.keys(): 
			prob_test = self.prob_set(N,V,all_grams[key],c_test)
			prob_train = self.prob_set(N,V,all_grams[key],c_train)

			count += prob_test*math.log(float(prob_test)/prob_train)

			# KL_array.append(prob_test*math.log(float(prob_test)/prob_train))

		print(count)

		return 1




	def __init__(self,file_in_domain, file_out_domain):
		count_without_caps = different_words(file_in_domain,'test')[1]
		self.in_domain = count_without_caps
		count_without_caps = different_words(file_out_domain,'test')[1]
		self.out_domain = count_without_caps

		print(self.KL_divergence(self.in_domain,self.out_domain))

		# number = self.number_chars_in_set(self.in_domain)
		# print(number)




	# def KL_divergency_3_grams(self):
	# 	for key in 


if __name__ == "__main__":

	files_3_dataset = ["ewt", "gum", "lines", "partut"]
	files_only_test = ["foot", "natdis", "pud"]
	type_of_set = ["train", "dev", "test"]

	grams = {}

	tel = {'jack': 1, 'sape': 1, 'apen':2}
	til = {'zuj': 2, 'ra': 5}

	n = Noisiness(files_only_test[2],files_only_test[0])
	# print(n.out_of_Vocabulary())
	# grams = n.generate_n_grams_char("perro",3)
	# grams = n.KL_divergency()

	# number = n.number_chars_in_set()
	# print(number)

	# print(sum(zip(tel.values(),til.values())))