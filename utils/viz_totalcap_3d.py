"""
    Functions to visualize human poses
    adapted from https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/viz.py
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import imageio
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from utils import forward_kinematics as fk
filenames = []
class Ax3DPose(object):
    def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c", label=['GT', 'Pred']):
        """
        Create a 3d pose visualizer that can be updated with new poses.

        Args
          ax: 3d axis to plot the 3d pose on
          lcolor: String. Colour for the left part of the body
          rcolor: String. Colour for the right part of the body
        """

        # Start and endpoints of our representation
        self.I = np.array([1, 2, 3, 4, 5, 6,  1,  1, 8, 9,  10, 12, 13, 14, 16, 17, 19, 20, 6 ,6  ]) - 1
        self.J = np.array([2, 3, 4, 5, 6, 7, 16, 19, 9, 10, 11, 13, 14, 15, 17, 18, 20, 21, 8 ,12 ]) - 1
        # Left / right indicator
        self.LR = np.array([ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1  ], dtype=bool)
        self.ax = ax

        vals = np.zeros((21, 3))

        # Make connection matrix
        self.plots = []
        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            if i == 0:
                self.plots.append(
                    self.ax.plot(x, z, y, lw=2, linestyle='--', c=rcolor if self.LR[i] else lcolor, label=label[0]))
            else:
                self.plots.append(self.ax.plot(x, y, z, lw=2, linestyle='--', c=rcolor if self.LR[i] else lcolor))

        self.plots_pred = []
        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            if i == 0:
                self.plots_pred.append(self.ax.plot(x, y, z, lw=2, c=rcolor if self.LR[i] else lcolor, label=label[1]))
            else:
                self.plots_pred.append(self.ax.plot(x, y, z, lw=2, c=rcolor if self.LR[i] else lcolor))

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        # self.ax.set_axis_off()
        # self.ax.axes.get_xaxis().set_visible(False)
        # self.axes.get_yaxis().set_visible(False)
        self.ax.legend(loc='lower left')
        self.ax.view_init(120, -90)

    def update(self, gt_channels, pred_channels):
        """
        Update the plotted 3d pose.

        Args
          channels: 96-dim long np array. The pose to plot.
          lcolor: String. Colour for the left part of the body.
          rcolor: String. Colour for the right part of the body.
        Returns
          Nothing. Simply updates the axis with the new pose.
        """

        assert gt_channels.size == 63, "channels should have 63 entries, it has %d instead" % gt_channels.size
        gt_vals = np.reshape(gt_channels, (21, -1))
        lcolor = "#8e8e8e"
        rcolor = "#383838"
        for i in np.arange(len(self.I)):
            x = np.array([gt_vals[self.I[i], 0], gt_vals[self.J[i], 0]])
            y = np.array([gt_vals[self.I[i], 1], gt_vals[self.J[i], 1]])
            z = np.array([gt_vals[self.I[i], 2], gt_vals[self.J[i], 2]])
            self.plots[i][0].set_xdata(x)
            self.plots[i][0].set_ydata(y)
            self.plots[i][0].set_3d_properties(z)
            self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)
            # self.plots[i][0].set_alpha(0.5)

        assert pred_channels.size == 63, "channels should have 63 entries, it has %d instead" % pred_channels.size
        pred_vals = np.reshape(pred_channels, (21, -1))
        lcolor = "#9b59b6"
        rcolor = "#2ecc71"
        for i in np.arange(len(self.I)):
            x = np.array([pred_vals[self.I[i], 0], pred_vals[self.J[i], 0]])
            y = np.array([pred_vals[self.I[i], 1], pred_vals[self.J[i], 1]])
            z = np.array([pred_vals[self.I[i], 2], pred_vals[self.J[i], 2]])
            self.plots_pred[i][0].set_xdata(x)
            self.plots_pred[i][0].set_ydata(y)
            self.plots_pred[i][0].set_3d_properties(z)
            self.plots_pred[i][0].set_color(lcolor if self.LR[i] else rcolor)
            # self.plots_pred[i][0].set_alpha(0.7)

        r = 30
        xroot, yroot, zroot = gt_vals[0, 0], gt_vals[0, 1], gt_vals[0, 2]
        self.ax.set_xlim3d([-r + xroot, r + xroot])
        self.ax.set_zlim3d([-r + zroot, r + zroot])
        self.ax.set_ylim3d([-r + yroot, r + yroot])
        self.ax.set_aspect('auto')


def plot_predictions(gt_3d, pred_3d, ax, f_title,act,seq,input_n):
    # Load all the data
    # parent, offset, rotInd, expmapInd = fk._some_variables()

    nframes_pred = pred_3d.shape[0]

    pose_dim = list(range(0, 63))
    gt_3d= gt_3d[:,pose_dim]
    pred_3d= pred_3d[:,pose_dim]
    
    for i in range(input_n):
        pred_3d[i,:]=gt_3d[i,:]


    # === Plot and animate ===
    ob = Ax3DPose(ax)
    # Plot the prediction
    for i in range(nframes_pred):
        #print(gt_3d[i, :].cpu().numpy().shape)
        ob.update(gt_3d[i, :].cpu().detach().numpy(), pred_3d[i, :].cpu().detach().numpy())
        ax.set_title(f_title + '   Frame:{:d}  '.format(i + 1), loc="left")
        plt.show(block=False)
        
        file_path ='./vis_result/{}/Sequence_{}/'.format(act,seq)
        isExist = os.path.exists(file_path)
        if not isExist:
            os.mkdir(file_path)

        
        
        filename = './vis_result/{}/Sequence_{}/'.format(act,seq) + f_title + 'Frame_{:d}.png'.format(i+1)
        filenames.append(filename)
        
        plt.savefig(filename)
        plt.pause(0.05)

    with imageio.get_writer('./vis_result/{}/Sequence_{}/'.format(act,seq) + f_title + '_result.gif' , mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

        # remove all the images
        # for filename in set(filenames):
        #      os.remove(filename)
            
    filenames.clear()
    

def show2Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False, GT=True):
    """Visualize a 2d skeleton
    Args
    channels: 64x1 vector. The pose to plot.
    ax: matplotlib axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
    Returns
    Nothing. Draws on ax.
    """

    assert channels.size == 63, "channels should have 63 entries, it has %d instead" % channels.size
    vals = np.reshape( channels,(21, -1) )

    # Start and endpoints of our representation
    I = np.array([1, 2, 3, 4, 5, 6,  1,  1, 8, 9,  10, 12, 13, 14, 16, 17, 19, 20, 6 ,6  ]) - 1
    J = np.array([2, 3, 4, 5, 6, 7, 16, 19, 9, 10, 11, 13, 14, 15, 17, 18, 20, 21, 8 ,12 ]) - 1
    # Left / right indicator
    LR = np.array([ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1  ], dtype=bool)


    # Make connection matrix
    for i in np.arange( len(I) ):
        x, y = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(2)]
        if GT==False:
            ax.plot(x, y, lw=2, c=lcolor if LR[i] else rcolor)
        else:
            ax.plot(x, y, lw=2,linestyle='--', c=lcolor if LR[i] else rcolor)

    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Get rid of tick labels
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])

    RADIUS = 30 # space around the subject
    xroot, yroot = vals[0,0], vals[0,1]
    ax.set_xlim([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim([-RADIUS+yroot, RADIUS+yroot])
    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("z")

    ax.set_aspect('equal')


def plot_predictions_2d(gt_3d, pred_3d, f_title,act,seq,input_n):
    # Load all the data
    # parent, offset, rotInd, expmapInd = fk._some_variables()
    
    gs1 = gridspec.GridSpec(1, 9) # 1 rows, 9 columns
    gs1.update(wspace=-0.00, hspace=0.12) # set the spacing between axes.
    subplot_idx, exidx = 1, 0
    
    pose_dim = list(range(0, 63))
    gt_3d= gt_3d[:,pose_dim]
    pred_3d= pred_3d[:,pose_dim]
    
    for i in range(input_n):
        pred_3d[i,:]=gt_3d[i,:]
    
    showed_frame = [0, 3, 6, 9, 12, 15, 18, 21, 24]

    # === Plot 2D pose ===
    for i  in range(len(showed_frame)):
        ax = plt.subplot(gs1[subplot_idx-1])
        g2d = gt_3d[showed_frame[i],:].cpu().detach().numpy()
        show2Dpose(g2d,ax,"#8e8e8e","#383838",GT=True)
        p2d = pred_3d[showed_frame[i],:].cpu().detach().numpy()
        show2Dpose(p2d,ax,"#9b59b6","#2ecc71",GT=False)
        ax.set_title('   Frame:{:d}  '.format(showed_frame[i]+1), loc="center",fontdict={'fontsize':50})
        plt.show(block=False)
        
        #exidx = exidx + 1
        subplot_idx = subplot_idx + 1


    filename = './vis_result/{}/Sequence_{}/'.format(act,seq) + f_title + '2d.png'.format(i+1)
    filenames.append(filename)
    
    plt.savefig(filename)
    plt.pause(0.05)
            
    filenames.clear()

    