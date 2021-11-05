from sys import _xoptions
import cv2
import numpy as np 
import math
import matplotlib.pyplot as plt



class HoverUtils:
    def __init__(self):
        self.mkHighlight = False
        self.plot = None
        self.xPoints = None
        self.yPoints = None

    def setHighlight(self, val):
        self.mkHighlight = val

    def getHighlight(self):
        return self.mkHighlight
    
    def setPlot(self, plt):
        self.plot = plt 

    def getPlot(self):
        return self.plot


    def setData(self, xPoints, yPoints):
        self.xPoints = xPoints
        self.yPoints = yPoints

    def hoverImage(self, event, fig, annotBox, xybox, pointIndexList, im, images_dict):
            plot = self.getPlot()
            data = plot.get_offsets()
            xCurr = data[:, 0]
            yCurr = data[:, 1]
            isHighlighted = self.getHighlight()
            if plot.contains(event)[0] and isHighlighted:
                ind, = plot.contains(event)[1]["ind"]
                # get the figure size
                w,h = fig.get_size_inches()*fig.dpi
                ws = (event.x > w/2.)*-1 + (event.x <= w/2.) 
                hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
                # if event occurs in the top or right quadrant of the figure,
                # change the annotation box position relative to mouse.
                annotBox.xybox = (xybox[0]*ws, xybox[1]*hs)
                # make annotation box visible
                annotBox.set_visible(True)
                # place it at the position of the hovered scatter point
                x = xCurr[ind]
                y = yCurr[ind]
                annotBox.xy =(x, y)
                # set the image corresponding to that point
                x = round(x, 4)
                y = round(y, 4)
                idx = findIndex((x,y), pointIndexList)
                im.set_data(images_dict[idx])
            else:
                #if the mouse is not over a scatter point
                annotBox.set_visible(False)
            fig.canvas.draw_idle()
    
    def highlight(self, _,highlightButton, ax, toHoverOn, colorMap, categories, edgeColors):
            xPoints = self.xPoints
            yPoints = self.yPoints
            currPlot = self.getPlot()
            checked = highlightButton.get_status()[0]
            currPlot.remove()
            if checked:
                plot = ax.scatter(xPoints[toHoverOn].squeeze(),
                    yPoints[toHoverOn].squeeze(),
                    c=colorMap[categories[toHoverOn]],
                    alpha=0.7, edgecolors=edgeColors, s=100, linewidths=2)
                self.setHighlight(True)
                self.setPlot(plot)
            else:
                plot = ax.scatter(xPoints.squeeze(),
                    yPoints.squeeze(),
                    c=colorMap[categories],
                    alpha=0.7, edgecolors=edgeColors, s=100, linewidths=2)
                self.setHighlight(False)
                self.setPlot(plot)
            plt.draw()


#Returns:   1. dictionary where an index points to corresponding point
#           2. List of tuples, where first element is a point (x,y) and second element is an index
def loadHighlighted(indices, imageRefs, path, xPoints, yPoints, imgSizeHover=(40,40)):
    names = [imageRefs[i] for i in indices]
    ref_images = [cv2.imread(f"{path}/{img_name}") for img_name in names]
    ref_images = [cv2.cvtColor(img, cv2.cv2.COLOR_RGB2GRAY) for img in ref_images]
    ref_images = [cv2.resize(img, dsize=imgSizeHover, interpolation=cv2.cv2.INTER_CUBIC) for img in ref_images]
    assert any(img is None for img in ref_images) == False, "Loading image references was not succesful, check path"
    ref_images = np.array(ref_images)
    assert ref_images.ndim == 3, "Something wrong with image references"

    ref_images_dict = dict()
    tupleToIndex = []
    #convert to dictionary because we need to access by indices when hovering
    for i, img in enumerate(ref_images):
        x = round(xPoints[indices[i]], 4)
        y = round(yPoints[indices[i]], 4)
        ref_images_dict[indices[i]] = img
        tupleToIndex.append(([x,y], indices[i]))

    return ref_images_dict, tupleToIndex


#Calculates the n samples which to hover on based on some function
#Currently hover points are decided by whether color (object) of gt and learned representation matches 
#and how far they are distant from cluster center
#NOTE: dists is samples x cluster_centers
def hoverPicker(dists, calc_color=None, gt_color=None, n=100):
    #calc bool array where labeling (i.e color) is different)
    if calc_color is None or gt_color is None:
        diffLabelInd = list(range(0, dists.shape[0]))
    else:
        hasSameLabel = np.invert(np.min(np.isclose(calc_color, gt_color), axis=1))
        diffLabelInd = hasSameLabel.nonzero()[0]
        print("---------------")
        print(f"There are {round((len(diffLabelInd) / dists.shape[0]), 2)}% samples that have incorrect object label")
        print(f"{n} samples will be avaible for further inspection")
        print("---------------")
    

    dist_to_own_center = np.min(dists, axis=1)
    dist_to_own_center = dist_to_own_center[diffLabelInd]
    #sort in descending order and pick n indices
    sorted_dist_ind = dist_to_own_center[::-1].argsort()[:n]
    

    return sorted_dist_ind

#Find for a point (x,y) the corresponding index in list of tuples where each tuple element
#contains a point and index 
def findIndex(point, lstOfTuplesWIdx):
    x = point[0]
    y = point[1]
    for element in lstOfTuplesWIdx:
        currPt = element[0]
        xCurr = currPt[0]
        yCurr = currPt[1]
        idx = element[1]

        if math.isclose(xCurr, x, rel_tol=1e-5) and math.isclose(yCurr, y, rel_tol=1e-5):
            return idx
    raise ValueError("No matching index found")