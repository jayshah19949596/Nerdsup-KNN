import numpy as np
from collections import Counter
from sklearn.manifold import TSNE
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
output_notebook()


def load_data(file):
	return np.loadtxt(file)


class KNN(object):
	def __init__(self, train_data, test_data, k):
		self.train_data = train_data[:, :-1]
		self.train_label = train_data[:, -1]
		self.test_data = test_data[:, :-1]
		self.test_label = test_data[:, -1]
		self.k = k
		self.correct = 0
		
	def normalize(self):
		mean = np.mean(self.train_data, axis=0)
		std_dev = np.std(self.train_data, axis=0)
		for i in range(0, self.train_data.shape[1]):
			self.train_data[:, i] = self.train_data[:, i] - mean[i]
			self.train_data[:, i] = self.train_data[:, i] / std_dev[i]
			self.test_data[:, i] = self.test_data[:, i] - mean[i]
			self.test_data[:, i] = self.test_data[:, i] / std_dev[i]
		# print(self.test_data[0, 0])
	
	def classify(self):
		for i in range(0, len(self.test_data)):
			distance = calculate_distance(self.test_data[i, :], self.train_data)
			distance = np.concatenate([[distance, self.train_label]])
			distance = distance.T
			distance = np.array(sorted(distance, key=lambda x: x[0]))
			self.show_results(distance, i)
		self.show_accuracy()
		
	def show_results(self, distance, row_number):
		if self.k == 1:
			k_neighbours = distance[self.k, :]
			predicted = k_neighbours[1]
			true = self.test_label[row_number]
			
		elif self.k > 1:
			k_neighbours = distance[0: self.k, :]
			counts = Counter(k_neighbours[:, 1])
			predicted = counts.most_common(1)[0][0]
			true = self.test_label[row_number]
			
		if true == predicted:
			self.correct += 1
			print("correct prediction = ", self.correct)
		
	def show_accuracy(self):
		print("accuracy = ", (self.correct/self.test_label.shape[0]))
			

def calculate_distance(x, y):
	distance = x - y
	distance = np.square(distance)
	distance = np.sum(distance, axis=1)
	distance = np.sqrt(distance)
	return distance


k_value = 1
train_file = "pendigits_training.txt"
test_file = "pendigits_test.txt"
train = load_data(train_file)
test = load_data(test_file)
knn = KNN(train, test, k_value)
knn.normalize()
knn.classify()

tsne = TSNE(n_components=2, random_state=0)
words_top_ted_tsne = tsne.fit_transform(knn.test_data)

p = figure(tools="pan,wheel_zoom,reset,save",
		   toolbar_location="above",
           title="vector T-SNE for most polarized words")

words_to_visualize = list([])
words_to_visualize.append("Jai")

colors_list = list([])
colors_list.append("#00ff00")
source = ColumnDataSource(data=dict(x1=words_top_ted_tsne[:, 0],
                                    x2=words_top_ted_tsne[:, 1],
                                    names=words_to_visualize,
                                    color=colors_list))

p.scatter(x="x1", y="x2", size=8, source=source, fill_color="color")

word_labels = LabelSet(x="x1", y="x2", text="names", y_offset=6,
                  text_font_size="8pt", text_color="#555555",
                  source=source, text_align='center')

p.add_layout(word_labels)

print("show")
show(p)
