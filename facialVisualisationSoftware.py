from mainClass import videoClass

# videClass params = (visualisation type, video file, stats color, circle size, sensitivity tuples (dist/angle), landmark divsions)
# landmarkDivision = [jaw, eyebrow, nose, eye, mouth]

# s = videoClass("Landmark", "test1.flv", (255,255,255), "S", (20, 7), (7, 6), (15, 10), (7, 8), (5, 5), landmarkDivision=[2,3,2,2,4])
s = videoClass("Point", "test2.flv", (255,255,255), "S", (20, 10), (10, 3), (40, 15), (10, 6), (6, 7), landmarkDivision=[2,3,2,2,4])
