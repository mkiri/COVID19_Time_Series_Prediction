import numpy as np
from multiprocessing.pool import ThreadPool
from pyspark.ml.tuning import _parallelFitTasks


class RollingKFoldCV(CrossValidator):
    """
    Modified CrossValidator to allow rolling k-fold cross validation using an increasing window. 
    Aimed to prevent data leakage associated with time series data (can't predict past using future data).
    """
        
    def _fit(self, dataset):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        nFolds = self.getOrDefault(self.numFolds)
        seed = self.getOrDefault(self.seed)
        metrics = [0.0] * numModels

        # Use rolling window instead of random folds
        rowNumCol = self.uid + "_rownum"
        w = Window().orderBy(lit('A')) # Dummy window to create row number
        df = dataset.select("*", row_number().over(w).alias(rowNumCol))
        h = df.count()/(nFolds+1)

        pool = ThreadPool(processes=self.getParallelism())
        subModels = None
        collectSubModelsParam = self.getCollectSubModels()
        if collectSubModelsParam:
            subModels = [[None for j in range(numModels)] for i in range(nFolds)]

        for i in range(nFolds):
            # Get rolling (increasing) window
            validateLB = (i + 1) * h
            validateUB = (i + 2) * h
            validation = df.filter((df[rowNumCol] >= validateLB) & (df[rowNumCol] < validateUB)).cache()
            train = df.filter(df[rowNumCol] < validateLB).cache()

            tasks = _parallelFitTasks(est, train, eva, validation, epm, collectSubModelsParam)
            for j, metric, subModel in pool.imap_unordered(lambda f: f(), tasks):
                metrics[j] += (metric / nFolds)
                if collectSubModelsParam:
                    subModels[i][j] = subModel

            validation.unpersist()
            train.unpersist()

        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)
        bestModel = est.fit(dataset, epm[bestIndex])
        return self._copyValues(CrossValidatorModel(bestModel, metrics, subModels))