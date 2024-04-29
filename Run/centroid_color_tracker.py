from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
    def __init__(self, maxDisappeared=25, maxColorDistance=50):
        # Initialize the next unique object ID and dictionaries for tracking objects and disappearances
        self.nextObjectID = 0
        self.objects = OrderedDict()  # This now stores tuples (centroid, color)
        self.disappeared = OrderedDict()

        # Maximum consecutive frames an object can disappear before deregistration
        self.maxDisappeared = maxDisappeared
        self.maxColorDistance = maxColorDistance  # Maximum color distance to consider object the same

    def register(self, centroid, color):
        # Register a new object with its centroid and color
        self.objects[self.nextObjectID] = (centroid, color)
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # Deregister an object by ID
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects, colors):
        # Check for empty rectangle list
        if len(rects) == 0:
            for objectID in self.disappeared.keys():
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        # Process input centroids and colors
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i], colors[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroidsColors = list(self.objects.values())
            objectCentroids = np.array([t[0] for t in objectCentroidsColors])
            objectColors = np.array([t[1] for t in objectCentroidsColors])

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            C = dist.cdist(np.array(objectColors), np.array(colors), metric='euclidean')
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                colorDistance = C[row, col]
                if colorDistance <= self.maxColorDistance:
                    self.objects[objectID] = (inputCentroids[col], colors[col])
                    self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(D.shape[0])).difference(usedRows)
            unusedCols = set(range(D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], colors[col])

        return self.objects
