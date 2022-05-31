from myxgboost import MyXgboost
x_train = []
x_test = []
y_train = []
target = []

my_model = MyXgboost()
my_model.fit(x_train, y_train)
y_test = my_model.test(x_test)
score = compare(y_test, target)