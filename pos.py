import json

def different_words(file, type_of_set):
	set_to_count = json.load(open("./corpus/en/en."+file+"."+type_of_set+".json"))
	count = {}
	count_without_caps = {}
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

	#print(count)
	print("Tamano count: ", len(count))
	print("Tamano count sin caps: ", len(count_without_caps))




if __name__ == "__main__":
	files_3_dataset = ["ewt", "gum", "lines", "partut"]
	files_only_test = ["foot", "natdis", "pud"]
	type_of_set = ["train", "dev", "test"]
	different_words(files_3_dataset[0], type_of_set[0])
