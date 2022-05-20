# -*- coding: utf-8 -*-

class Elasticnet:
        def Elasticnet(data, label):
                algorithm = Elasticnet.Elasticnet(data, 'BMI')
                algorithm.extract_labels()
                X_train, X_test, y_train, y_test = algorithm.split_data()
                elastic_model = algorithm.train_model(X_train, y_train)
                predictions = algorithm.predict(elastic_model, X_test)
                model, cv = algorithm.define_model()
                scores = algorithm.evaluate_model(model, cv)
                metrics = Metrics.Metrics(y_test, predictions)
                r2 = metrics.r_squared()
                mse = metrics.mean_squared_error()
                mae = metrics.mean_absolute_error()
                rmse = metrics.root_mean_squared_error()
                print("Input file: %s" % args.f)

                # Data preparation for plotting(should become a class)
                scores1 = copy(scores)
                scores2 = {"Algorithm1": scores, "Algorithm2": scores1}
                #
                visuals = Visualisations.Visualisations(scores2, cv)
                visuals.boxplot()