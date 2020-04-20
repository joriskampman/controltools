"""
This shadows the analysis in python of Vigney's work on the long shaft
"""

# import numpy as np
# import matplotlib.pyplot as plt
# import control as ct
import os
import auxtools as aux
import numpy as np
import logtools as lt
import control as cm
from scipy.io import loadmat
from warnings import warn  # noqa
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pdb  # noqa
import sys


class RootLocus(object):
  """
  The root-locus class which can not only display the root-locus for a system, but also has some
  slight interactivity
  """
  # plot properties
  min_markersize = 0.5
  max_markersize = 40

  ctrl_jump_perc = 2.5
  nof_K = 2001
  K_start_pow10 = -2
  K_end_pow10 = 8
  sfmax_K = 2.
  plot_pz = True
  plot_ref = False

  nof_zeta_lines = 11
  olines = None
  otxts = None
  Hz = True
  K_base = 1.
  iselect = None

  def __init__(self, G, C=1., Plot=True, order_of_coef_to_check=0, is_relative=False,
               K_ref=1., **kwargs):
    # Define the feedback system with a PI as C

    # make forward path (connect C and G)
    # preprocess plant to become siso and get generic input and outputnames
    if not G.issiso():
      raise ValueError("The input plant (G) must be a SISO system!")

    if isinstance(G, cm.TransferFunction):
      G = cm.tf2ss(G)

    if isinstance(C, (float, int)):
      C = cm.tf2ss(cm.tf(float(C), 1.))
    elif isinstance(C, cm.TransferFunction):
      C = cm.tf2ss(C)

    self.G = G
    self.C = C
    self.dt = G.dt

    for key, val in kwargs.items():
      setattr(self, key, val)

    if Plot:
      self.plot(order_of_coef_to_check=order_of_coef_to_check,
                is_relative=is_relative,
                K_ref=K_ref)
    else:
      self.calc_roots()

  def on_pick(self, event):
    """
    lkjasdf
    """
    if event.mouseevent.button == 3:  # remove selection
      self.clear_selected_points()
    elif event.mouseevent.button == 1:
      self.iselect = (np.mean(event.ind) + 0.5).astype(int)
      self.plot_selected_points()

  def on_view_change(self, event):
    """
    view is changed, change zeta axes
    """
    self.plot_zeta_lines(is_update=True)

  def on_key_press(self, event):
    """
    adsfasdf
    """
    if self.iselect is None:
      return

    if event.key == 'up':
      self.iselect += 1
    elif event.key == 'down':
      self.iselect -= 1
    elif event.key == 'ctrl+up':
      self.iselect += np.fmax(1, np.int(self.ctrl_jump_perc*self.nof_K/100))
    elif event.key == 'ctrl+down':
      self.iselect -= np.fmax(1, np.int(self.ctrl_jump_perc*self.nof_K/100))
    elif event.key == 'end':
      self.iselect = self.nof_K - 1
    elif event.key == 'home':
      self.iselect = 0
    elif event.key in ('delete', 'backspace'):
      self.clear_selected_points()
    else:
      return

    # prevent out of bounds errors
    if self.iselect < 0:
      self.iselect = 0
    elif self.iselect >= self.nof_K:
      self.iselect = self.nof_K - 1

    self.plot_selected_points()

  def clear_selected_points(self):
    """
    clear the selected points
    """
    self.iselect = None
    for clart in self.clarts:
      clart.set_visible(False)
    self.ax.set_title(None)
    self.fig.tight_layout()
    plt.draw()

  def plot_selected_points(self):
    """
    plot or update the selected points
    """
    # get data for all rl's for this K value
    dcays_arr, worfs_arr = split_s_plane_coords(self.rootsarr, self)
    K = self.Ks[self.iselect]
    sigmas_this_K = dcays_arr[self.iselect, :]
    worfs_this_K = worfs_arr[self.iselect, :]

    # start the new title string
    titlestr = "<< Closed-loop is "
    if np.alltrue(sigmas_this_K < 0.):
      titlestr += r"$\mathbf{STABLE}$"
      titlecolor = 'k'
    else:
      titlestr += r"$\mathbf{UNSTABLE}$"
      titlecolor = 'r'
    titlestr += " >>\n"
    titlestr += "Gain (K) = {:0.1e}".format(K)

    # loop all root-locus plots
    for ipole in range(dcays_arr.shape[1]):

      # get the numbers
      sigma = sigmas_this_K[ipole]
      worf = worfs_this_K[ipole]

      # show the clarts
      self.clarts[ipole].set_xdata(sigma)
      self.clarts[ipole].set_ydata(worf)
      self.clarts[ipole].set_visible(True)

      # derive zeta
      zeta = np.cos(np.arctan(worf/-sigma))

      # append to title string
      if self.Hz:
        worf_str_part = "$f_d$ = {:0.4f} [Hz]".format(worf)
      else:
        worf_str_part = "$\\omega_d$ = {:0.4f} [rad/s]".format(worf)
      titlestr += ("\n({:d}) $\\zeta$ = {:0.2f}, $\\sigma$ =  {:0.2f}, {:s}".
                   format(ipole, zeta, sigma, worf_str_part))

    self.ax.set_title(titlestr, color=titlecolor, fontsize=8)
    self.fig.tight_layout()
    plt.draw()
    plt.pause(1e-6)

  def calc_zeta_line_thetas(self):
    """
    define the zeta line theta angles (which are the angles relative to the negative x axis)
    """
    ax = self.ax

    thetas = []
    xlims = aux.arrayify(ax.get_xlim())
    ylims = aux.arrayify(ax.get_ylim())

    for x in xlims:
      for y in ylims:
        # calculate angle theta
        theta = np.arctan2(y, -x)
        thetas.append(theta)

    thetas = np.array(thetas)
    thetas[np.isclose(thetas, np.pi)] = 0.

    thetas[thetas > np.pi/2] = np.pi/2
    thetas[thetas < -np.pi/2] = -np.pi/2

    # get the minimum theta for this window and the maximum
    theta_min = thetas.min()
    theta_max = thetas.max()

    # define thetas -> discard corner points after linspace'ing
    thetas = np.array(np.linspace(theta_min, theta_max, self.nof_zeta_lines))

    return thetas

  def plot_zeta_lines(self, is_update=False):
    """
    plot the lines of constant zeta (damping ratio) values
    """
    # now divide this section in plot angles
    ax = self.ax
    xlims = aux.arrayify(ax.get_xlim())
    ylims = aux.arrayify(ax.get_ylim())
    if not is_update:
      self.olines = [None]*self.nof_zeta_lines
      self.otxts = [None]*self.nof_zeta_lines

    # recalculate the thetas to display
    disp_thetas = self.calc_zeta_line_thetas()

    for iline, disp_theta in enumerate(disp_thetas[1:-1]):
      rc = np.tan(np.pi - disp_theta)
      # calculate points for the 4 limits (2x x, 2x y)
      if np.isclose(np.pi/2, np.abs(disp_theta)):
        xs_line = aux.arrayify((0., 0.))
        if disp_theta < 0.:
          ys_line = aux.arrayify([0., ylims[0]])
        else:
          ys_line = aux.arrayify([0, ylims[1]])
      else:
        ys_at_xlims = rc*xlims
        xs_at_ylims = ylims/rc

        xs_lines_all = np.concatenate((xlims, xs_at_ylims))
        ys_lines_all = np.concatenate((ys_at_xlims, ylims))

        # keep the points that are in the limits
        is_valid_x = (xs_lines_all >= xlims[0])*(xs_lines_all <= xlims[1])
        is_valid_y = (ys_lines_all >= ylims[0])*(ys_lines_all <= ylims[1])
        is_valid_pt = is_valid_x*is_valid_y

        xs_line = xs_lines_all[is_valid_pt]
        ys_line = ys_lines_all[is_valid_pt]

      # plot only the part in the LHP
      is_in_rhp = xs_line > 0.0001
      ys_line[is_in_rhp] = 0.
      xs_line[is_in_rhp] = 0.

      # is exit at xlimit?
      xtxt = xs_line.mean()
      ytxt = ys_line.mean()

      # plot the line1
      if is_update:
        self.olines[iline].set_xdata(xs_line)
        self.olines[iline].set_ydata(ys_line)
      else:
        oline, = ax.plot(xs_line, ys_line, 'k:', linewidth=1, zorder=1)
        self.olines[iline] = oline

      # calculate the damping ratio (zeta)
      zeta = np.cos(disp_theta)
      txt = r"$\zeta$ = {:0.2f}".format(zeta)
      # txt = "{:0.2f}".format(zeta)

      trans_angle = (180. +
                     ax.transData.transform_angles(np.array((np.rad2deg(np.pi - disp_theta),)),
                                                   np.array([xtxt, ytxt]).reshape((1, 2)))[0])

      if is_update:
        self.otxts[iline].set_position((xtxt, ytxt))
        self.otxts[iline].set_text(txt)
        self.otxts[iline].set_rotation(trans_angle)
      else:
        self.otxts[iline] = ax.text(xtxt, ytxt, txt, ha='center', va='center',
                                    fontweight='bold', fontsize='x-small', rotation=trans_angle,
                                    rotation_mode='anchor', backgroundcolor='w',
                                    bbox=dict(facecolor='w', boxstyle='round', pad=0.3))

    self.fig.tight_layout()
    plt.draw()
    plt.show(block=False)
    self.fig.tight_layout()
    plt.draw()

  def create_augmented_C(self, order_of_coef_to_check):
    """
    iorder2check is the coefficient of the s order to check
    """
    C_num = cm.ss2tf(self.C).num[0][0]
    C_den = cm.ss2tf(self.C).den[0][0]

    K_base = C_num[-1 - order_of_coef_to_check]

    C_aug = make_block('tfdata', 'Ctq_aug', dt=self.dt, num=C_num/K_base, den=C_den,
                       keep_names=True)

    self.C_aug = C_aug
    self.K_base = K_base

  def calc_roots(self, order_of_coef_to_check=1, is_relative=False):
    """
    calculate the roots -> called on plotting
    """
    # augment C to check K for a single coefficient
    self.create_augmented_C(order_of_coef_to_check)

    if is_relative:
      pows_K = np.linspace(-1, 1., self.nof_K)
      sfs_K = self.sfmax_K**pows_K
      Ks = self.K_base*sfs_K
    else:
      Ks = np.logspace(self.K_start_pow10, self.K_end_pow10, self.nof_K)

    CG_aug = build_system([self.C_aug, self.G], tag="CG_aug", prune=True).minreal()

    # get poles and zeros (from minreal system)
    poles = CG_aug.pole()
    zeros = CG_aug.zero()
    nof_zeros = zeros.size
    nof_poles = poles.size
    nof_roots = np.fmax(nof_zeros, nof_poles)

    nof_infs = nof_roots - np.fmin(nof_zeros, nof_poles)

    # calculate P and Q from 1 + KCG = 1 + k(Q/P) =0
    P = cm.ss2tf(CG_aug).den[0][0]
    Q = np.zeros_like(P)  # init to have the same dimensions as P
    Q_ = cm.ss2tf(CG_aug).num[0][0]
    Q[nof_infs:] = Q_

    # initialize the roots
    # loop all K values (slow, but for now that's OK)
    rootsarr = np.empty((self.nof_K, nof_roots), dtype=np.complex_)
    prev_roots = np.roots(P)
    for ik, K in enumerate(Ks):
      fchar = P + K*Q
      roots = np.roots(fchar)
      # check the order of the roots

      # roots may change position -> correct for this
      # 1. make delta matrix and find the minimum per column
      deltas = np.abs(prev_roots.reshape(-1, 1) - roots.reshape(1, -1))
      indices = np.argmin(deltas, axis=1)

      # in case double indices occur -> make random choice via noise matrix
      # while loop necessary in case noise matrix is not decisive
      while np.diff(indices).prod() == 0:
        # doesn't matter, pick one via random permutation
        indices = np.argmin(deltas*np.random.random((nof_roots, nof_roots)), axis=1)

      # add these to the list
      rootsarr[ik, :] = roots[indices]
      prev_roots = rootsarr[ik, :]

    # find which locus is going to infinity
    inotinfs = np.r_[:nof_roots]
    if not is_relative:
      deltas_zeros = np.abs(roots[indices].reshape(-1, 1) - zeros.reshape(1, -1))
      mindist_to_zeros = np.min(deltas_zeros, axis=1)
      isort = np.argsort(mindist_to_zeros)[-1::-1]  # invert to sort from large to small
      iinfs = isort[:nof_infs]
      inotinfs = np.setdiff1d(np.r_[:nof_roots], iinfs)

    # add stuff to this object
    self.poles = poles
    self.zeros = zeros
    self.nof_roots = nof_roots
    self.Ks = Ks
    self.rootsarr = rootsarr
    self.inotinfs = inotinfs

    return split_s_plane_coords(rootsarr, Hz=self.Hz)

  def plot(self, order_of_coef_to_check=1, is_relative=False, K_ref=1.):
    '''
    plot the root-locus
    '''
    if is_relative:
      iref = self.nof_K//2
      plot_ref = True
      plot_pz = False
    else:
      plot_ref = False
      plot_pz = True

    dcays_arr, worfs_arr = self.calc_roots(order_of_coef_to_check=order_of_coef_to_check,
                                           is_relative=is_relative)

    nof_poles = self.poles.size
    nof_zeros = self.zeros.size

    nof_roots = np.fmax(nof_poles, nof_zeros)

    # determine x and y limits
    xlims = aux.arrayify(aux.bracket(dcays_arr[:, self.inotinfs]))
    ylims = aux.arrayify(aux.bracket(worfs_arr[:, self.inotinfs]))

    dists = np.abs(np.diff(self.rootsarr, axis=0))
    cdists = np.cumsum(dists, axis=0)
    cdistsn = cdists/cdists[-1, :]
    cdistsn_ = np.vstack((np.zeros((1, nof_poles), dtype=np.float_), cdistsn))
    sizearr = aux.data_scaling(cdistsn_, self.min_markersize, self.max_markersize, func='linear')

    self.fig = plt.figure(aux.figname("root-locus"))
    self.ax = self.fig.add_subplot(111)
    self.ax.grid(True)

    colors = aux.jetmod(nof_roots, 'vector', bright=True)
    iipoles = np.argmin(np.abs(self.rootsarr[0, :].reshape(-1, 1) - self.poles.reshape(1, -1)),
                        axis=0)
    iizeros = np.argmin(np.abs(self.rootsarr[-1, :].reshape(-1, 1) - self.zeros.reshape(1, -1)),
                        axis=0)
    clarts = [None]*nof_roots
    for iroot, (dcays, worfs) in enumerate(zip(dcays_arr.T, worfs_arr.T)):
      is_valid_x = (dcays >= xlims[0])*(dcays <= xlims[1])
      is_valid_y = (worfs >= ylims[0])*(worfs <= ylims[1])
      is_valid = is_valid_x*is_valid_y

      colvec = aux.color_vector(self.nof_K, colors[iroot, :], os=0.35)
      cmap = ListedColormap(colvec, name="root_locus_p{:d}".format(iroot))
      self.ax.plot(dcays[is_valid], worfs[is_valid], '-', color=colors[iroot, :], linewidth=0.5,
                   picker=False, zorder=1)
      self.ax.scatter(dcays[is_valid], worfs[is_valid], s=sizearr[is_valid, iroot],
                      c=sizearr[is_valid, iroot], cmap=cmap, marker='o', picker=True,
                      label="pole {:d}".format(iroot), zorder=3)

      if plot_ref:
        self.ax.plot(dcays_arr[iref, iroot], worfs_arr[iref, iroot], '*', mew=2, mfc='none',
                     markersize=20, mec=colors[iroot, :], zorder=2)
      # plot poles and zeros if flag is on
      if plot_pz:
        self.ax.plot(*split_s_plane_coords(self.poles[iipoles[iroot]], Hz=self.Hz), 'x', mew=2,
                     mfc='none', markersize=15, mec=colors[iroot, :], zorder=2)
      # plot clicked on artists (clarts)
      clarts[iroot], = self.ax.plot(0., 0., 's', mec='k', mfc='none', markersize=15, mew=3,
                                    visible=False, zorder=10)

    if plot_pz:
      for izero in range(nof_zeros):
        self.ax.plot(*split_s_plane_coords(self.zeros[izero], Hz=self.Hz), 'o', mew=2,
                     mfc='none', markersize=15, mec=colors[iizeros[izero], :], zorder=3)

    # add the zeta lines
    self.plot_zeta_lines(False)
    self.clarts = clarts
    self.is_update = True

    self.cid1 = self.fig.canvas.mpl_connect('pick_event', self.on_pick)
    self.cid2 = self.ax.callbacks.connect('xlim_changed', self.on_view_change)
    self.cid3 = self.ax.callbacks.connect('ylim_changed', self.on_view_change)
    self.cid4 = self.fig.canvas.mpl_connect('resize_event', self.on_view_change)
    self.cid5 = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    plt.show(block=False)
    plt.draw()


def plot_state_space_matrices(ss, display_type='compressed', show_values=True, zero_marker=None,
                              show_names=True, split_plots=False, aspect='auto', cmap=None,
                              suptitle=None, maximize=True):
  """
  plot all 4 state-space matrices including enumerated ticks

  arguments:
  ----------
  ss : control.StateSpace object
       The state-space objects for which the figure(s) has(have) to be made
  display_type : [ 'normal', 'binary', 'posneg', 'compressed'], default='normal'
             A plot type which indicates how the matrices should be plotted. Options are:
             - 'normal': just the values as is
             - 'binary': boolean value to indicate whether a value is zero or nonzero
             - 'posneg': gives a color depending on whether it is negative, zero or positive
             - 'compressed': uses a hyperbolic tangent to compress the values to a range of [-1, 1]
  show_values : boolean, default=False
                Flag to indicate whether to display the TRUE values in the matrices. The format
                will be the `general` ({:g}) format
  zero_marker : [ None | str ], default=None
                Determines how to display the zeroes in the matrix. None will display nothing and
                in case of a string, every zero value will be marked as provided.
                Note that `show_values` and `zero_marker` work independently
  show_names: bool, default=True
              flag to indicate whether to show the names (state, input, output) as tick labels
  split_plots : bool, default=False
                flag to indicate whether to split the plots or to make one big 2x2 subplot figure
  aspect: ['auto' | 'equal'], default='auto'
          Aspect ratio flag, is passed to the call of *imshow*
  cmap : [ None | ... ], default=None
         Colormap instance or None. In case `None`, the function will decide the colormap based
         om the display_type. The value if not None is passed to the *imshow* function as kwarg.
  suptitle : [None | str ], default=None
             Figure's suptitle, passed to the figure

  Returns:
  --------
  fig, ax : [tuple of tuples | tuple]
            Depending on the *split_plots* flag the output is either a tuple of tuples
            (one per figure), or a tuple of the figure and the axes
  """
  def _labelmaker(fmtstr, nameslist):
    """
    subfunction to make the labels
    """
    lablist = [fmtstr.format(str_, idx) for idx, str_ in enumerate(nameslist)]
    return lablist

  div_cmap = 'bwr'
  bin_cmap = 'traffic_light'

  matdict = dict()
  matdict['A'] = dict(show_x_labels=split_plots,
                      show_y_labels=True,
                      rowlabels=_labelmaker("d({:s})/dt - {:2d}", ss.statenames),
                      collabels=_labelmaker("{:s} - {:2d}", ss.statenames))
  matdict['B'] = dict(show_x_labels=split_plots,
                      show_y_labels=split_plots,
                      rowlabels=_labelmaker("d({:s})/dt - {:2d}", ss.statenames),
                      collabels=_labelmaker("{:s} - {:2d}", ss.inputnames))
  matdict['C'] = dict(show_x_labels=True,
                      show_y_labels=True,
                      rowlabels=_labelmaker("{:s} - {:2d}", ss.outputnames),
                      collabels=_labelmaker("{:s} - {:2d}", ss.statenames))
  matdict['D'] = dict(show_x_labels=True,
                      show_y_labels=split_plots,
                      rowlabels=_labelmaker("{:s} - {:2d}", ss.outputnames),
                      collabels=_labelmaker("{:s} - {:2d}", ss.inputnames))

  if not isinstance(ss, cm.StateSpace):
    raise ValueError("The input argument `ss` is not of type cm.StateSpace")

  # determine the colormap
  kwargs = dict(aspect=aspect, cmap=cmap)
  if cmap is None:
    kwargs['cmap'] = div_cmap
    if display_type == 'binary':
      kwargs['cmap'] = bin_cmap
  else:
    kwargs['cmap'] = div_cmap

  if not split_plots:
    fig, axs = plt.subplots(2, 2, num=aux.figname("State-space overview"), sharex=False,
                            sharey=False)
    if maximize:
      mng = plt.get_current_fig_manager()
      mng.window.showMaximized()

  for iplot, (key, mdict) in enumerate(matdict.items()):
    if split_plots:
      fig, ax = plt.subplots(1, 1, num=aux.figname("{:s} matrix".format(key)))
      if maximize:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
      fig.tight_layout()

    else:
      ax = axs[np.unravel_index(iplot, (2, 2))]

    # get matrix data
    matdata = getattr(ss, key)
    if display_type == 'normal':
      matdisp = matdata.copy()
      minval = matdisp.min()
      maxval = matdisp.max()
      maxext = np.fmax(np.abs(minval), np.abs(maxval))
      kwargs.update(vmin=-maxext, vmax=maxext)
    elif display_type == 'binary':
      matdisp = np.int_(~np.isclose(0., matdata))
    elif display_type == 'compressed':
      matdisp = np.tanh(matdata)
      kwargs.update(vmin=-1., vmax=1.)
    elif display_type == 'posneg':
      matdisp = np.sign(matdata)
    else:
      raise ValueError("The *display_type* keyword argument value ({}) is not valid.".
                       format(display_type))

    nr, nc = matdata.shape

    ax.imshow(matdisp, **kwargs)

    # set title, grid and labels
    ax.set_title("{:s} [{:d} x {:d}]".format(key, nr, nc))
    ax.set_xticks(np.r_[:nc])
    ax.set_xticklabels(ax.get_xticks(), fontsize=7)
    if mdict['show_x_labels']:
      ax.set_xticklabels(mdict['collabels'], fontsize=7, rotation=45, va='top', ha='right')
    ax.set_yticks(np.r_[:nr])
    ax.set_yticklabels(ax.get_yticks(), fontsize=7)
    if mdict['show_y_labels']:
      ax.set_yticklabels(mdict['rowlabels'], fontsize=7)
    ax.tick_params(axis='both', which='major', length=0)

    # make minor ticks for dividing lines
    ax.set_xticks(np.r_[-0.5:nc+0.5:1], minor=True)
    ax.set_yticks(np.r_[-0.5:nr+0.5:1], minor=True)

    ax.grid(which='minor', linewidth=1)

    # show the values in the matrix
    if show_values:
      for irow in range(nr):
        for icol in range(nc):
          if not np.isclose(matdata[irow, icol], 0.):
            ax.text(icol, irow, "{:.1g}".format(matdata[irow, icol]), fontsize=6, ha='center',
                    va='center', clip_on=True, bbox={'boxstyle':'square',
                                                     'pad':0.0,
                                                     'facecolor': 'none',
                                                     'lw': 0.,
                                                     'clip_on': True})

    if zero_marker is not None:
      rcs = np.argwhere(np.isclose(0., matdata))
      ax.scatter(rcs[:, 1], rcs[:, 0], c='k', marker=zero_marker, s=15)

    if split_plots:
      plt.show(block=False)
      plt.draw()
      plt.pause(1e-1)
      plt.tight_layout()
      plt.draw()
      plt.pause(1e-6)

  if not split_plots:
    plt.show(block=False)
    plt.draw()
    plt.pause(1e-1)
    plt.tight_layout()
    plt.draw()
    plt.pause(1e-6)


def print_overview(Hsys, file=None):
  """
  print an overview of the (sub)system/block
  """
  if file is None:
    file = sys.stdout
  elif isinstance(file, str):
    file = open(file, "wt")

  print("==========================================================================", file=file)
  print("===============  (sub)system overview  ===================================", file=file)
  print("==========================================================================", file=file)

  print("Tag: {:s}".format(Hsys.tag), file=file)
  print("Is plant: {}".format(Hsys.is_plant), file=file)
  print("Is SISO: {}".format(Hsys.issiso()), file=file)
  print("Is discrete time: {}".format(Hsys.isdtime()), file=file)
  print("Is continuous time: {}".format(Hsys.isctime()), file=file)
  print("\nNumber of states: {:2d}".format(Hsys.states), file=file)
  print("--------------------", file=file)
  for istate, state in enumerate(Hsys.statenames):
    print("  [{:d}] {:s}".format(istate, state), file=file)

  print("\nNumber of inputs: {:2d}".format(Hsys.inputs), file=file)
  print("--------------------", file=file)
  for iin, input_ in enumerate(Hsys.inputnames):
    print("  [{:d}] {:s}".format(iin, input_), file=file)

  print("\nNumber of outputs: {:2d}".format(Hsys.outputs), file=file)
  print("---------------------", file=file)
  for iout, output in enumerate(Hsys.outputnames):
    print("  [{:d}] {:s}".format(iout, output), file=file)


def _linear_interpolation(ys, xs, ythres):
  """
  adsfadsfasdf
  """
  dx = np.diff(xs).item()
  dy = np.abs(np.diff(ys).item())
  dy0 = np.abs(ys[0] - ythres)

  dyratio = dy0/dy

  t_interp = xs[0] + dx*dyratio

  return t_interp


def settling_time(ys, ts=None, yfinal=None, perc=2., nbuffer=20, interp='linear'):
  """
  calculate the setting time based on the step response output
  """
  frac = perc/100.

  if ts is None:
    ts = np.r_[:len(ys)]

  if yfinal is None:
    yfinal = np.mean(ys[-nbuffer:])

  ythres = perc/100
  if np.isclose(yfinal/1000, 0.):
    ythres = ys.max()*(perc/100)

  # check if last req_buffer smaples are within perc
  if np.max(np.abs(ys[-nbuffer:] - yfinal)) > ythres:
    warn('Shit has not settled upto {:d} samples before the end.'.format(nbuffer))
    return np.nan, np.nan

  # check 2: 
  yrefinv = ys[-1::-1]  # from right to left

  if np.isclose(yfinal, 0.):
    frac = ys.max()*frac

  i_last_unsettled = ys.size - np.argwhere(np.abs(yrefinv - yfinal) > frac).item(0) - 1
  if interp == 'linear':

    # determine threshold
    sgn = np.sign((ys[i_last_unsettled] - yfinal)/frac)
    ythres = yfinal + sgn*frac

    # linear interpolation
    t_settle = _linear_interpolation(ys[i_last_unsettled:i_last_unsettled+2],
                                     ts[i_last_unsettled:i_last_unsettled+2],
                                     ythres)
  elif interp == 'nearest':
    t_settle = ts[i_last_unsettled + 1]
  else:
    raise NotImplementedError("The interpolation scheme provided ({}) is not implemented".
                              format(interp))

  return t_settle, yfinal


def rise_time(ys, ts=None, yfinal=None, percs=[10., 90.], nbuffer=20, interp='linear'):
  """
  calculate the rise time; the time between 10 and 90 of the final value
  """
  if yfinal is None:
    yfinal = np.mean(ys[-nbuffer:])

  fracs = [perc/100 for perc in percs]

  # start fraction passed for the first time
  i_low = np.argwhere(ys > fracs[0]*yfinal).item(0) - 1
  i_high = np.argwhere(ys > fracs[1]*yfinal).item(0) - 1

  dt_low = _linear_interpolation(ys[i_low:i_low+2], ts[i_low:i_low+2], fracs[0]*yfinal)
  dt_high = _linear_interpolation(ys[i_high:i_high+2], ts[i_high:i_high+2], fracs[1]*yfinal)

  t_rise = dt_high - dt_low

  return t_rise


def peak_chars(ys, ts=None):
  """
  find the time of the peak
  """
  if ts is None:
    ts = np.r_[:ys.size]

  imax = np.argmax(ys)
  ienv = imax + np.r_[-1:2]

  # fit quadratic
  a, b, c = np.polyfit(ts[ienv], ys[ienv], 2)

  # peak at -b/2a
  t_peak = -b/(2*a)

  # y_peak
  y_peak = np.polyval((a, b, c), t_peak)

  return t_peak, y_peak


def settling_min(ys, ts=None):
  """
  find the settling min time
  """
  if ts is None:
    ts = np.r_[:ys.size]

  imax = np.argmax(ys).item()

  # find the minimum value after the peak
  ys_ = ys[imax:]
  imin = imax + np.argmin(ys_).item()

  ienv = imin + np.r_[-1:2]

  a, b, c = np.polyfit(ts[ienv], ys[ienv], 2)

  t_dip = -b/(2*a)

  y_dip = np.polyval((a, b, c), t_dip)

  return t_dip, y_dip


def overshoot(ys, ts=None, yfinal=None, nbuffer=20, as_percentage=True):
  """
  calculate the percentage overshoot
  """
  if ts is None:
    ts = np.r_[:ys.size]

  if yfinal is None:
    yfinal = np.mean(ys[-nbuffer:])

  # get peak
  tpeak, ypeak = peak_chars(ys, ts=ts)

  outval = ypeak - yfinal
  if as_percentage:
    outval = 100*outval/yfinal

  return outval


def stepinfo(ys, ts=None, yfinal=None, rise_percs=[10., 90.], settle_perc=2., interp='linear',
             nbuffer=20):
  """
  gather all step response information available
  """
  if ts is None:
    ts = np.r_[:ys.size]

  if yfinal is None:
    yfinal = np.mean(ys[-nbuffer:])

  # rise_time
  trise = rise_time(ys, ts=ts, yfinal=yfinal, percs=rise_percs, interp=interp)

  # settling time
  tsettle = settling_time(ys, ts=ts, yfinal=yfinal, perc=settle_perc, interp=interp)

  # peak chars
  tpeak, ymax_settle = peak_chars(ys, ts=ts)

  # minimum settle
  tmin_settle, ymin_settle = settling_min(ys, ts)

  # overshoot
  perc_overshoot = overshoot(ys, ts=ts, yfinal=yfinal, as_percentage=True)

  # place them all in a dict
  out = dict(rise_time=trise,
             settling_time=tsettle,
             peak_time=tpeak,
             peak_value=ymax_settle,
             percentage_overshoot=perc_overshoot,
             minimum_settling_time=tmin_settle,
             minimum_settling_value=ymin_settle,
             final_value=yfinal,
             nbuffer=nbuffer,
             rise_percs=rise_percs,
             settling_percentage=settle_perc)

  return out


def _prune(Hsys, strlist, inout, keep_or_prune):
  """
  prune the inputs of a system
  """
  strlist = aux.tuplify(strlist)
  if inout == 'in':
    allnames = Hsys.inputnames.copy()
  elif inout == 'out':
    allnames = Hsys.outputnames.copy()

  indices = []
  for i2k in strlist:
    ifnd = aux.find_elm_containing_substrs(i2k, allnames, nreq=1, strmatch="all")
    indices.append(ifnd)

  if keep_or_prune == 'keep':
    indices2keep = indices
  else:
    indices2keep = np.setdiff1d(np.r_[:len(allnames)], indices)

  # set all invalids to zero
  if inout == 'in':
    Hsys.B = Hsys.B[:, indices2keep]
    Hsys.D = Hsys.D[:, indices2keep]
    Hsys.inputnames = aux.listify(aux.arrayify(Hsys.inputnames)[indices2keep])
    Hsys.inputs = len(indices2keep)
  elif inout == 'out':
    Hsys.C = Hsys.C[indices2keep, :]
    Hsys.D = Hsys.D[indices2keep, :]
    Hsys.outputnames = aux.listify(aux.arrayify(Hsys.outputnames)[indices2keep])
    Hsys.outputs = len(indices2keep)

  return indices2keep


def prune_ios(Hsys, inputnames=None, outputnames=None, keep_or_prune='keep'):
  """
  prune the inputs and outputs from a system
  """
  iins = np.r_[:Hsys.inputs]
  iouts = np.r_[:Hsys.outputs]
  if inputnames is not None:
    iins = _prune(Hsys, inputnames, 'in', keep_or_prune)

  if outputnames is not None:
    iouts = _prune(Hsys, outputnames, 'out', keep_or_prune)

  return iins, iouts


def build_system(blocks, Qstrings=None, tag="system", opens=None, shorts=None, prune=False):
  """
  build_system(blocks, Qstrings[, opens, shorts])

  build the system from blocks (make_block) and define the connections in a [(from, to)]
  array-like of tuples.

  If one of the box has the flag `is_plant` set to True, the wind speed, azimuth and reference
  generator and rotor speeds are taken from that

  Arguments:
  ----------
  blocks : array-like of blocks
           contains blocks that are made by `make_block`. They need to have attributes
           `inputnames` and `outputnames` as well as a tag as a minimum. `make_block`
           ensures the creation of those
  tag : str, default='system'
        The id tag the returned system will have
  Qstrings : None or array-like of tuples of strings, default=None
             Contains tuples of strings describing the `from` and `to` of any signal. This is
             different from the order in the control module Q array (which has inputs first,
             and then outputs). I've found it more easy to visualize it from output to input.

             Every input/output string is split into parts and searched in the list of inputs/
             outputs.

             the inputs may be a tuple too, this means that the output will be split and fed into
             multiple inputs.

             Note: a `!` as first character indicates a NEGATIVE feedback
  opens : [None | str | array-like of strings], default=None
          Defines the blocks which are to be made into and open; that is a zero output or a
          broken connection
  shorts : [None | str | array-like of strings], default=None
           Defines the blocks which are a shortcut; output = input

  Returns:
  --------
  out : state-space object
  """
  blocks = aux.listify(blocks)

  # next step: define interconnections based on input and outputnames
  if Qstrings is None:
    # connect all outputs to all subsequent inputs, works only for siso systems
    Qstrings = []
    for iblock in range(len(blocks)-1):
      block_out = blocks[iblock]
      block_in = blocks[iblock + 1]
      if not block_out.issiso():
        raise ValueError("In case Qstrings=None, all blocks must be SISO." +
                         "block with tag `{}` is not!".format(block_out.tag))

      Qstrings.append((block_out.outputnames[0], block_in.inputnames[0]))

  Qstrings_arr = np.array(Qstrings, dtype=[('outputs', object), ('inputs', object)])

  opens = aux.tuplify(opens)
  shorts = aux.tuplify(shorts)

  for iblock in range(len(blocks)):
    btag = blocks[iblock].tag
    if btag in opens:
      blocks[iblock] = make_block('open', btag)

    if btag in shorts:
      blocks[iblock] = make_block('short', btag)

  # 1: check all blocks mentioned in Q
  sys_inputnames = []
  sys_outputnames = []
  iplant = -1
  for iblock, block in enumerate(blocks):
    btag = block.tag
    sys_inputnames += block.inputnames.copy()
    sys_outputnames += block.outputnames.copy()
    if block.is_plant:
      iplant = iblock

  Ql = []
  for qa in Qstrings_arr:
    out = qa['outputs']
    # find exact output index (remember: start at 1!!!!)
    iout = aux.find_elm_containing_substrs(out, sys_outputnames, nreq=1, strmatch='all')

    ins = aux.tuplify(qa['inputs'])
    for in_ in ins:
      # check if it contains an exclamation point (!) indicating a negative feedback
      is_fb = False
      if in_.startswith('!'):
        is_fb = True
        in_ = in_[1:]

      # find the index
      iin_ = aux.find_elm_containing_substrs(in_, sys_inputnames, nreq=1, strmatch='all')

      # add to Q array
      Ql.append((iin_+1, (1 - 2*is_fb)*(iout+1)))

  Q = aux.arrayify(Ql)

  inputv = np.r_[:len(sys_inputnames)] + 1
  outputv = np.r_[:len(sys_outputnames)] + 1

  Hsys_parts = cm.append(*blocks)
  Hsys = cm.connect(Hsys_parts, Q, inputv, outputv)

  if iplant > -1:
    Hsys.is_plant = True
  Hsys.tag = tag
  Hsys.inputnames = sys_inputnames
  Hsys.outputnames = sys_outputnames

  if iplant > -1:
    possible_attrs = ('wind_speed',
                      'azimuth',
                      'ref_generator_speed',
                      'ref_generator_torque',
                      'ref_rotor_speed',
                      'ref_pitch')
    for attr in possible_attrs:
      if hasattr(blocks[iplant], attr):
        setattr(Hsys, attr, getattr(blocks[iplant], attr))

  if prune:
    # prune all that are connected (be carefull with this automated setting!)
    Qarr = aux.arrayify(Qstrings)
    prune_ios(Hsys, inputnames=Qarr[:, 1], outputnames=Qarr[:, 0], keep_or_prune='prune')

  return Hsys


def modify_plant(Hplant, what2mod, replace=None, rename=None, reorder=None, remove=None):
  """
  modify part of the plant
  """
  if not hasattr(Hplant, what2mod):
    raise ValueError("The plant does not have an attribute `{}`".format(what2mod))

  dimdict = dict.fromkeys(['A', 'B', 'C', 'D'], [])
  if what2mod == 'states':
    nameattr = 'statenames'
    dimdict['A'] = [0, 1]
    dimdict['B'] = [0]
    dimdict['C'] = [1]
  elif what2mod == 'inputs':
    nameattr = 'inputnames'
    dimdict['B'] = [1]
    dimdict['D'] = [1]
  elif what2mod == 'outputs':
    nameattr = 'outputnames'
    dimdict['C'] = [0]
    dimdict['D'] = [0]

  # 1: removing states
  irem = aux.find_elm_containing_substrs(remove, getattr(Hplant, nameattr))
  ikeep = np.setdiff1d(np.r_[:getattr(Hplant, what2mod)], irem)

  # update states/inputs/outputs
  setattr(Hplant, what2mod, ikeep.size)
  # update names
  setattr(Hplant, nameattr, getattr(Hplant, nameattr)[ikeep])

  # update matrices
  for matid in dimdict.keys():
    mdata = getattr(Hplant, matid)
    if 0 in dimdict[matid]:
      mdata = mdata[ikeep, :]
    if 1 in dimdict[matid]:
      mdata = mdata[:, ikeep]

    # set modified matrix to Hplant
    setattr(Hplant, matid, mdata)

  # 2: replace and rename
  setattr(Hplant, nameattr,
          aux.arrayify(aux.modify_strings(getattr(Hplant, nameattr), globs=replace, specs=rename)))

  # 3: reorder
  if reorder is not None:
    nof_elms = getattr(Hplant, what2mod)
    if len(reorder) != nof_elms:
      raise ValueError("The number of elements in `reorder` ({:d}) is not matching the number of".
                       format(len(reorder)) + " outputs ({:d}".format(nof_elms))

    isort = np.ones((nof_elms), dtype=int)
    for ielm, str_ in enumerate(reorder):
      isort[ielm] = aux.find_elm_containing_substrs(str_, getattr(Hplant, nameattr), nreq=1,
                                                    raise_except=False, strmatch='all')

    # update matrices
    for matid in dimdict.keys():
      mdata = getattr(Hplant, matid)
      if 0 in dimdict[matid]:
        mdata = mdata[isort, :]
      if 1 in dimdict[matid]:
        mdata = mdata[:, isort]

      # set modified matrix to Hplant
      setattr(Hplant, matid, mdata)

    setattr(Hplant, nameattr, getattr(Hplant, nameattr)[isort])


def make_block(btype, tag, dt=0., inames=None, onames=None, force_statespace=True,
               keep_names=False, **blargs):
  """
  to be filled in
  """
  # take blargs as keyword arguments for the subfunctions
  remove_useless = False

  is_plant = False
  if btype == 'plant':
    if 'ss' in blargs.keys():
      block = blargs['ss']
    elif 'ssdata' in blargs.keys():
      block = cm.StateSpace(**blargs, remove_useless=remove_useless)
    elif 'tf' in blargs.keys():
      block = cm.tf2ss(blargs['tf'])
    is_plant = True
    if inames is None:
      if hasattr(block, 'inputnames'):
        inames = block.inputnames.copy()
    if onames is None:
      if hasattr(block, 'outputnames'):
        onames = block.outputnames.copy()
  elif btype == 'g':
    block = cm.StateSpace([], [], [], blargs['gain'], remove_useless=remove_useless)
  elif btype == 'short':
    block = cm.StateSpace([], [], [], 1., remove_useless=remove_useless)
  elif btype == 'open':
    block = cm.StateSpace([], [], [], 0., remove_useless=remove_useless)
  elif btype == 'inv':
    block = cm.StateSpace([], [], [], -1., remove_useless=remove_useless)
  elif btype == 'ss':
    block = blargs['ss']
  elif btype == 'ssdata':
    block = cm.StateSpace(**blargs, remove_useless=remove_useless)
  elif btype == 'tfdata':
    block = cm.tf(blargs['num'], blargs['den'])
    if force_statespace:
      block = cm.tf2ss(block)
  elif btype == 'tf':
    block = blargs['tf']
    if force_statespace:
      block = cm.tf2ss(block)
  elif btype in ['p', 'i', 'd', 'pi', 'pd', 'pid']:
    block = pid(**blargs)
  elif btype == 'nf':
    block = notch(**blargs)
  elif btype == 'bp':
    block = bandpass(**blargs)
  elif btype == 'lp':
    block = low_pass_filter(**blargs)
  elif btype == 'hp':
    block = high_pass_filter(**blargs)
  else:
    raise NotImplementedError("The wanted btype ({}) is not implemented".format(btype))

  block.dt = dt
  block.tag = tag
  block.is_plant = is_plant

  gen_inputnames = True
  gen_outputnames = True
  if keep_names:
    if hasattr(block, 'inputnames'):
      gen_inputnames = False
    if hasattr(block, 'outputnames'):
      gen_outputnames = False

  if gen_inputnames:
    # set inputnames and outputnames
    block.inputnames = []
    nins = block.inputs
    for iin in range(nins):
      block.inputnames.append("{:s}_in{:d}".format(tag, iin))
      if inames is not None:
        given_name = inames[iin].replace(' ', '_')
        block.inputnames[-1] += "_{:s}".format(given_name)
  if gen_outputnames:
    block.outputnames = []
    nouts = block.outputs
    for iout in range(nouts):
      block.outputnames.append("{:s}_out{:d}".format(tag, iout))
      if onames is not None:
        given_name = onames[iout].replace(' ', '_')
        block.outputnames[-1] += "_{:s}".format(given_name)

  return block


def load_models(files, dirname='', states_to_ignore=[], vwinds_wanted=None, azis_wanted=None,
                dt=0.):
  """
  get a list of plant linearizations
  """
  files = aux.listify(files)

  Hplants_list = []
  for file in files:
    Hplants, vwinds, azis = load_state_space_model_file(file, dirname=dirname,
                                                        states_to_ignore=states_to_ignore, dt=dt)
    if vwinds_wanted is None:
      vwinds_wanted = np.copy(vwinds)
    else:
      vwinds_wanted = aux.listify(vwinds_wanted)

    if azis_wanted is None:
      azis_wanted = azis.copy()
    else:
      azis_wanted = aux.listify(azis_wanted)

    for vwind_wanted in vwinds_wanted:
      iwind = aux.get_closest_index(vwind_wanted, vwinds)
      for azi_wanted in azis_wanted:
        iazi = aux.get_closest_index(azi_wanted, azis)

        Hplant = Hplants[iwind, iazi]
        Hplant.is_plant = True
        Hplant.tag = 'plant_w{:0.1f}_a{:0.1f}'.format(vwind_wanted, azi_wanted)
        Hplants_list.append(Hplant)

  if len(Hplants_list) == 1:
    Hplants_list = Hplants_list[0]

  return Hplants_list


def plot_multiple_bodes(Hplants_list, inputname, outputname, split=False, show_nyquist=False):
  """
  plot multiple siso bodes
  """
  fig = None
  axs = None
  colors = aux.jetmod(len(Hplants_list), 'vector', bright=True)
  for color, Hplant in zip(colors, Hplants_list):
    iin = aux.substr2index(inputname, Hplant.inputnames)
    iout = aux.substr2index(outputname, Hplant.outputnames)
    label = "wind = {:0.1f}, azi = {:0.1f} [deg] ({})".format(Hplant.wind_speed,
                                                              np.rad2deg(Hplant.azimuth),
                                                              Hplant.filename)
    if split:
      fig = None
      axs = None
    fig, axs, lines, texts = plot_bode(Hplant, iin, iout, fig=fig, axs=axs, color=color,
                                       show_legend=True, label=label,
                                       show_nyquist=show_nyquist)

  return fig, axs


def _handle_iin_iout(iin, iout, Hplant):
  """
  handle different forms of iin, iout (str, array, whatever)
  """

  # do check on iin and iout
  if isinstance(iin, str):
    iin = aux.find_elm_containing_substrs(tuple(iin.split()), Hplant.inputnames, nreq=1)

  if isinstance(iout, str):
    iout = aux.find_elm_containing_substrs(tuple(iout.split()), Hplant.outputnames, nreq=1)

  return iin, iout


def plot_bode(Hplant, input_=0, output=0, omega_limits=[1e-3, 1e2], omega_num=1e4, dB=True,
              Hz=True, show_margins=False, show_nyquist=False, fig=None, axs=None, color='b',
              linestyle='-', show_legend=True, label=None, figname="Bode plots",
              suptitle=None):
  """
  plot a single bode plot
  """
  iin, iout = _handle_iin_iout(input_, output, Hplant)

  if figname is None:
    figname = suptitle

  # plot all possible transfer functions
  if fig is None:
    fig = plt.figure(aux.figname(figname))
    if show_nyquist:
      gs = fig.add_gridspec(2, 2)
    else:
      gs = fig.add_gridspec(2, 1)
    axs = [fig.add_subplot(gs[0, 0])]
    axs.append(fig.add_subplot(gs[1, 0], sharex=axs[0]))
    fig.suptitle(suptitle, fontweight="bold", fontsize=12)
    axs[0].set_title("Gain/Magnitude")
    if Hz:
      axs[0].set_xlabel("Frequency [Hz]")
    else:
      axs[0].set_xlabel("Angular frequency [rad/s]")
    axs[0].set_ylabel("magnitude [dB]")
    axs[0].grid(True)
    axs[0].axhline(0, color='k', linestyle=':')
    axs[1].set_title("Phase")
    if Hz:
      axs[1].set_xlabel("Frequency [Hz]")
    else:
      axs[1].set_xlabel("angular frequency [rad/s]")
    axs[1].set_ylabel("Phase [deg]")
    axs[1].grid(True)
    axs[1].axhline(-180, color='k', linestyle=':')

    if show_nyquist:
      axs.append(fig.add_subplot(gs[:, 1]))
      axs[2].set_title("Nyquist plot")
      axs[2].grid(True)
      # plot nyquist
      axs[2].axvline(color='k', linestyle=':')
      axs[2].axhline(color='k', linestyle=':')
      axs[2].plot(-1, 0, 'mo', markersize=6)

  if hasattr(Hplant, 'inputnames'):
    namein = Hplant.inputnames[iin]
  else:
    namein = "input_{:2d}".format(iin)

  if hasattr(Hplant, 'outputnames'):
    nameout = Hplant.outputnames[iout]
  else:
    nameout = "output_{:2d}".format(iout)

  if Hplant.issiso():
    Hsiso_this = Hplant
  else:
    Hsiso_this = make_siso(Hplant, iin, iout)

  mags, phs, omegas = cm.bode_plot(Hsiso_this, dB=dB, Hz=Hz, Plot=False,
                                   omega_limits=omega_limits, omega_num=np.int(omega_num))

  # convert phs to interval -inf, 0 to be able to verify the phase margins correctly
  if label is None:
    label = "{} -> {}".format(namein, nameout)

  if Hz:
    axs[0].semilogx(omegas/(2*np.pi), cm.mag2db(mags), "-", label=label, color=color,
                    linestyle=linestyle)
    axs[1].semilogx(omegas/(2*np.pi), np.rad2deg(phs), "-", label=label,
                    color=color, linestyle=linestyle)
  else:
    axs[0].semilogx(omegas, cm.mag2db(mags), "-", label=label, color=color,
                    linestyle=linestyle)
    axs[1].semilogx(omegas, np.rad2deg(phs), "-", label=label,
                    color=color, linestyle=linestyle)

  gm_, pm_, sm_, wg_, wp_, ws_ = cm.stability_margins(Hsiso_this, returnall=True)

  # plot gain margins
  for wg__, gm__ in zip(wg_, gm_):
    gmdb = cm.mag2db(gm__)
    if Hz:
      wg = w2f(wg__)
    else:
      wg = wg__
    if omega_limits[0] <= wg <= omega_limits[1]:
      axs[0].plot(wg*np.array([1., 1.]), [0., -gmdb], ':', color=color)
      axs[0].text(wg, -gmdb/2, "{:0.0f}".format(-gmdb), ha='center', va='center',
                  fontsize=7, fontweight='bold', color=color, backgroundcolor='w',
                  bbox={'pad': 0.1, 'color': 'w'})

  # phase margins
  for wp__, pm__ in zip(wp_, pm_):
    # check if it is relative to -180 or +180
    if Hz:
      wp = wp__/(2*np.pi)
    else:
      wp = wp__

    if omega_limits[0] <= wp <= omega_limits[1]:
      iomega = aux.get_closest_index(wp__, omegas, suppress_warnings=True)

      # offset = -(nof_folds*2. + 1.)*180.
      ytxt = (phs[iomega] - phs[iomega])/2.
      ypos = [np.rad2deg(phs[iomega]), -180.]

      ytxt = np.mean(ypos)
      axs[1].plot(wp*np.array([1., 1.]), ypos, ':', color=color)
      axs[1].text(wp, ytxt, "{:0.0f}".format(pm__),
                  ha='center', va='center', fontsize=7, fontweight='bold', color=color,
                  bbox={'pad': 0.1, 'color': 'w'})

  if show_nyquist:
    nqi, nqq, nqf = cm.nyquist_plot(Hsiso_this, omega=omegas, Plot=False)
    cvec = aux.color_vector(omegas.size, color, os=0.25)
    axs[2].scatter(nqi, nqq, s=2, c=cvec)

    nqii = np.interp(ws_, nqf, nqi)
    nqqi = np.interp(ws_, nqf, nqq)
    for ipt in range(nqii.size):
      axs[2].plot([-1., nqii[ipt]], [0., nqqi[ipt]], '-', color=color)

  if show_legend:
    axs[0].legend(fontsize=8, loc='lower left')

  plt.show(block=False)

  return fig, axs


def plot_single_plant_bodes(Hplant, inputs=None, outputs=None, omega_limits=[1e-3, 1e2],
                            omega_num=1e5, dB=True, Hz=True, show_margins=False,
                            show_nyquist=False, fig=None, axs=None, colors=None,
                            show_legend=True, figname=None, suptitle="Bode plots"):
  """
  generate all bode plots for a plant
  """

  # handle input_indices and output_indices
  if inputs is None:
    inputs = np.r_[0:Hplant.inputs]
  inputs = aux.arrayify(inputs)

  if outputs is None:
    outputs = np.r_[0:Hplant.outputs]
  outputs = aux.arrayify(outputs)

  # determine the colors
  nof_bodes = inputs.size*outputs.size
  if colors is None:
    colors = aux.jetmod(nof_bodes, 'vector', bright=True)

  iplot = -1
  for iin in inputs:
    for iout in outputs:
      iplot += 1
      (fig, axs) = plot_bode(Hplant, iin, iout, omega_limits=omega_limits,
                             omega_num=np.int_(omega_num), dB=dB, Hz=Hz, show_margins=show_margins,
                             show_nyquist=show_nyquist, fig=fig, axs=axs,
                             color=colors[iplot], linestyle='-', show_legend=show_legend,
                             figname=figname, suptitle=suptitle)

  return fig, axs


def plot_root_locus(Hplant, input_=0, output=0, use_ref=True, klims=[0.01, 100], num_k=1e3,
                    klist=None, isklog=True, figname=None, suptitle="Root-locus", ax=None,
                    legacy=False, title=None, Hz=False, show_legend=True, xlim=None, ylim=None):
  """
  wrapper around the root-locus plotting functions that returns a set of k's and loci
  """
  iin, iout = _handle_iin_iout(input_, output, Hplant)

  if figname is None:
    figname = suptitle

  if ax is None:
    fig = plt.figure(aux.figname(figname))
    ax = fig.add_subplot(111)

    fig.suptitle(suptitle, fontweight="bold", fontsize=12)
    ax.set_title(title)
    ax.set_xlabel(r"exponential decay rate ($\sigma$)")
    if Hz is True:
      ax.set_ylabel("frequency [Hz]")
    else:
      ax.set_ylabel(r"$angular velocity ($\omega$)")
    ax.grid(True)
  else:
    fig = ax.figure

  if title is None:
    title = "Root-locus for {:0.1e} <= K <= {:0.1e}".format(*klims)

  # can be only done for SISO systems (for now)
  if Hplant.issiso():
    Hsiso_this = Hplant
  else:
    Hsiso_this = make_siso(Hplant, iin, iout)

  if isklog:
    kvect = np.linspace(*klims, num_k)
  else:
    kvect = np.logspace(*np.log10(klims), num_k)

  # add a zero and inf to the list of kvect
  kvect = np.array([0., *kvect, 1e10], dtype=np.float_)

  if legacy:
    rlist, klist = cm.root_locus(Hsiso_this)
    return
  else:
    rlist, klist = cm.root_locus(Hsiso_this, kvect=kvect, Plot=False)

  # loop all starting poles (does it work when Nz > Hp)
  starting_poles = Hsiso_this.pole()
  colors = aux.jetmod(starting_poles.size, 'vector', bright=True)
  for ip, pole in enumerate(starting_poles):
    p_dcay0, p_worf0 = split_s_plane_coords(pole, Hz=Hz)

    label = r"pole @ $\sigma$={:0.1f}, ".format(p_worf0)
    if Hz:
      label += "f={:0.1f} [Hz]".format(p_dcay0)
    else:
      label += r"$\omega$={:0.1f} [rad/s]".format(p_worf0)

    # get locii
    dcays, worfs = split_s_plane_coords(rlist[:, ip])

    # plot starting pole
    # plot lines
    ax.plot(dcays[-1], worfs[-1], 'o', color=colors[ip, :], linewidth=3, mfc=colors[ip, :],
            mec=colors[ip, :], markersize=6)
    ax.plot(dcays[0], worfs[0], 'x', color=colors[ip, :], mew=3, markersize=10)
    ax.plot(dcays[1:-1], worfs[1:-1], '.-', color=colors[ip, :], linewidth=1)
    # asdf
    ax.plot(p_dcay0, p_worf0, 'x', markersize=10, color='k', mew=1)

  if show_legend:
    ax.legend(fontsize=8, loc='lower left')

  plt.show(block=False)

  return fig, ax


def find_valid_states(statenames, states_to_ignore):
  '''
  identify and return a TF array of the valid states
  '''
  # find the states that are valid
  is_valid_state = np.ones((statenames.size,), dtype=bool)
  if not states_to_ignore:
    return is_valid_state

  for igstate in states_to_ignore:
    is_valid_state_ = np.array([igstate.lower() not in state.lower() for state in statenames])
    is_valid_state *= is_valid_state_

  return is_valid_state


def _ignore_states_single(Hplant, states_to_ignore, howto):
  """
  ignore the states, `howto` determines if they are removed or zeroed
  """
  is_valid_state = find_valid_states(Hplant.statenames, states_to_ignore)
  if howto == 'remove':
    _remove_states_single(Hplant, is_valid_state)
  elif howto == 'zero':
    _zero_states_single(Hplant, ~is_valid_state)
  else:
    raise ValueError("The `howto` keyword value ({}) is not valid.".format(howto))


def _remove_states_single(Hplant, is_valid_state):
  """
  remove the states of a single plant
  """
  A = Hplant.A[is_valid_state, :]
  A = A[:, is_valid_state]
  B = Hplant.B[is_valid_state, :]
  C = Hplant.C[:, is_valid_state]
  Hplant.states = is_valid_state.sum()
  Hplant.statenames = Hplant.statenames[is_valid_state]
  Hplant.A = A
  Hplant.B = B
  Hplant.C = C


def _zero_states_single(Hplant, is_state_to_zero):
  """
  zero the states, but not remove them
  """
  Hplant.A[is_state_to_zero, :] = 0.
  Hplant.A[:, is_state_to_zero] = 0.
  Hplant.B[is_state_to_zero, :] = 0.
  Hplant.C[:, is_state_to_zero] = 0.

  Hplant.statenames = np.array([true*"[X] " + name + true*" [X]" for true, name in
                                zip(is_state_to_zero, Hplant.statenames)])


def ignore_states(Hplants, states_to_ignore):
  '''
  remove the unwanted states or single or multiple plants (i.e., a ndarray)
  '''
  if isinstance(Hplants, (list, np.ndarray)):
    for Hplant in Hplants:
      _ignore_states_single(Hplant, states_to_ignore)
  else:
    _ignore_states_single(Hplants, states_to_ignore)


def load_state_space_model_file(filename, dirname='.', states_to_ignore=[], dt=0,
                                remove_useless=False):
  """
  Load the state space model from a .mat file and return the transfer statespace model of the plant
  """
  ssmat = loadmat(os.path.join(dirname, filename))

  vwinds = ssmat['Windspeeds'].ravel()
  azis = ssmat['Azimuths'].ravel()

  # create state stapce model
  # get transfer function (must be homebrew function since it is a MIMO system)
  inputnames = aux.strip_all_spaces(ssmat['SYSTURB'][0, 0]['inputname'])
  outputnames = aux.strip_all_spaces(ssmat['SYSTURB'][0, 0]['outputname'])
  statenames = aux.strip_all_spaces(ssmat['SYSTURB'][0, 0]['statename'])

  Hplants = np.empty((vwinds.size, azis.size), dtype=np.object_)
  As = ssmat['SYSTURB'][0, 0]['A']
  Bs = ssmat['SYSTURB'][0, 0]['B']
  Cs = ssmat['SYSTURB'][0, 0]['C']
  Ds = ssmat['SYSTURB'][0, 0]['D']

  if azis.size == 1:
    # add extra dimension as placeholder
    As = np.expand_dims(As, -1)
    Bs = np.expand_dims(Bs, -1)
    Cs = np.expand_dims(Cs, -1)
    Ds = np.expand_dims(Ds, -1)
  if vwinds.size == 1:
    As = np.expand_dims(As, -1)
    Bs = np.expand_dims(Bs, -1)
    Cs = np.expand_dims(Cs, -1)
    Ds = np.expand_dims(Ds, -1)

    # roll this dimension to position 2
    As = np.swapaxes(As, 2, 3)
    Bs = np.swapaxes(Bs, 2, 3)
    Cs = np.swapaxes(Cs, 2, 3)
    Ds = np.swapaxes(Ds, 2, 3)

  for ivw, vwind in enumerate(vwinds):
    for iazi, azi in enumerate(azis):
      A = As[..., ivw, iazi]
      B = Bs[..., ivw, iazi]
      C = Cs[..., ivw, iazi]
      D = Ds[..., ivw, iazi]

      Hplant = cm.StateSpace(A, B, C, D, dt, remove_useless=remove_useless)
      Hplant.inputnames = inputnames
      Hplant.outputnames = outputnames
      Hplant.statenames = statenames
      Hplant.wind_speed = vwind
      Hplant.azimuth = azi
      Hplant.ref_generator_speed = ssmat['NomSpeedArray'][ivw, iazi]
      Hplant.ref_rotor_speed = ssmat['RotorSpeeds'][0, ivw]
      Hplant.ref_pitch = ssmat['PitchAngles'][ivw, iazi]
      Hplant.ref_generator_torque = ssmat['NomTorqueArray'][ivw, iazi]

      Hplant.dirname = dirname
      Hplant.filename = filename

      if len(states_to_ignore) > 0:
        _ignore_states_single(Hplant, states_to_ignore, 'zero')
      Hplants[ivw, iazi] = Hplant

  return Hplants, vwinds, azis


def make_siso(Hplant, input_, output, tag="siso"):
  """
  make a MIMO system into a SISO system
  """
  input_, output = _handle_iin_iout(input_, output, Hplant)

  siso = cm.rss(states=Hplant.states, inputs=1, outputs=1)
  siso.A = Hplant.A.copy()
  siso.B = Hplant.B[:, input_].reshape(-1, 1)
  siso.C = Hplant.C[output, :].reshape(1, -1)
  siso.D = Hplant.D[output, input_].reshape(1, 1)
  siso.is_plant = Hplant.is_plant
  siso.dt = Hplant.dt

  if hasattr(Hplant, 'inputnames'):
    siso.inputnames = aux.listify(Hplant.inputnames[input_])
  if hasattr(Hplant, 'outputnames'):
    siso.outputnames = aux.listify(Hplant.outputnames[output])
  if hasattr(Hplant, 'statenames'):
    siso.statenames = Hplant.statenames

  siso.tag = tag

  return siso


def _loadpj(items=None, dirname=None, filenames=None):
  '''
  shadow of loadpj.m
  '''
  def _select_file(dirname):

    filenames = aux.select_file(filetypes=[("Bladed simulation project", ".$TE"),
                                           ("All simulation BOW", ".$me")],
                                multiple=True,
                                title="Select the files to be loaded",
                                initialdir=dirname)
    if not filenames:
      raise ValueError("No file(s) selected. Function aborted")

    return filenames

  def _split_item_list(items_to_load):
    """
    split the items in the list of lists
    """
    description_list = []
    extension_list = []
    for item in items_to_load:
      description_list.append(item[0])
      extension_list.append(item[1])

    return description_list, extension_list

  items_to_load0 = np.array([['Performance_coeffs', '02'],
                             ['Power_curve', '03'],
                             ['Control_variables', '04'],
                             ['Drive_train', '05'],
                             ['Generator', '06'],
                             ['Summary', '07'],
                             ['Pitch_actuator', '08'],
                             ['Aero_B1', '09'],
                             ['Aero_B2', '10'],
                             ['Aero_B3', '11'],
                             ['Partial', '13'],
                             ['Environmental_information', '14'],
                             ['Blade_1_loads_Principal_axes', '15'],
                             ['Blade_2_loads_Principal_axes', '16'],
                             ['Blade_3_loads_Principal_axes', '17'],
                             ['Blade_deflections_1', '18'],
                             ['Blade_deflections_2', '19'],
                             ['Blade_deflections_3', '20'],
                             ['Hub_rotating', '22'],
                             ['Hub_fixed', '23'],
                             ['Yaw_bearing_loads', '24'],
                             ['Tower_loads', '25'],
                             ['Nacelle_motion', '26'],
                             ['External_controller', '29'],
                             ['Blade_absolute_position_1', '31'],
                             ['Blade_absolute_position_2', '32'],
                             ['Blade_absolute_position_3', '33'],
                             ['Blade_acceleration_1', '34'],
                             ['Blade_acceleration_2', '35'],
                             ['Blade_acceleration_3', '36'],
                             ['Blade_acceleration_3', '36'],
                             ['Hub', '37'],
                             ['Blade_1_loads_Root_axes', '41'],
                             ['Blade_2_loads_Root_axes', '42'],
                             ['Blade_3_loads_Root_axes', '43'],
                             ['Yaw_actuator', '44'],
                             ['Maximum PowerCoefficient', '55'],
                             ['Tower_displacements', '56'],
                             ['DataFromTurbine', '129'],
                             ['Structural_state_positions', '150'],
                             ['Pitch_Bearing', '151'],
                             ['Create_Control_Variables', '162'],
                             ['Whoever_knows', '101'],
                             ['Markov_001', '001']], dtype='U50')
  if not items:
    items_to_load = items_to_load0.tolist()
  else:
    if isinstance(items, str):
      items = [items,]

    items_to_load = []
    for item in items:
      if isinstance(item, str):
        # lookup the extension
        iitem = np.argwhere(items_to_load0[:, 0] == item)
        if iitem.size > 0:
          items_to_load.append([item, items_to_load0[iitem.item(), 1]])
      else:
        items_to_load.append(item)

  if not filenames:
    fname = _select_file(dirname)
  else:
    # tuple-ize in case given
    if isinstance(filenames, str):
      filenames =(os.path.join(dirname, filenames),)
    elif isinstance(filenames, (list, tuple)):  # list or tuple of values
      filenames = tuple([os.path.join(dirname, filename) for filename in filenames])

  lcdict = dict()
  for ilc, fname in enumerate(filenames):
    pjname = os.path.splitext(os.path.basename(fname))[0]
    path = os.path.dirname(fname)
    lcdict[ilc] = dict(loadcase=os.path.join(path, pjname))
    for iitem, item in enumerate(items_to_load):
      # load the header
      # note: [1] is the data only
      infodict, data = lt.BladedLoggings.read_logging(fname, ".%" + item[1], ".$" + item[1])

      # make time
      lcdict[ilc][item[0]] = dict(hdr=infodict,
                                  data=data)

  return lcdict


def notch(f0, damping_ratio, gain=1., fz=None, dt=0, force_statespace=True):
  """
  returns the transfer function nof a notch filter
  """
  omega0 = f2w(f0)
  if fz is None:
    omegaz = omega0
  else:
    omegaz = f2w(fz)

  num = [1, 0, omegaz**2]
  den = [1, 2*damping_ratio*omega0, omega0**2]

  out = gain*cm.tf(num, den, dt)

  if force_statespace:
    out = cm.tf2ss(out)

  return out


def bandpass(f0, damping_ratio, gain=1., dt=0, force_statespace=True):
  """
  create a bandpass filter
  """
  omega0 = f2w(f0)
  num = [2*damping_ratio*omega0, 0]
  den = [1., 2*damping_ratio*omega0, omega0**2]

  out = gain*cm.tf(num, den, dt)

  if force_statespace:
    out = cm.tf2ss(out)

  return out


def low_pass_filter(fc, gain=1., force_statespace=True, dt=0.):
  """
  define a n-th order low pass filter
  """
  omc = f2w(fc)
  Hlp = cm.tf([gain], [1./omc, 1.])
  if force_statespace:
    Hlp = cm.tf2ss(Hlp)

  return Hlp


def high_pass_filter(fc, gain=1., dt=0., force_statespace=True):
  """
  create a high pass filter
  """
  omc = f2w(fc)
  Hhp = cm.tf([gain/omc, 0], [1./omc, 1])

  if force_statespace:
    Hhp = cm.tf2ss(Hhp)

  return Hhp


def _handle_pid_inputs(**kwargs):
  """
  handle the inputs to the pid
  """
  # handle inputs whether times or gains and convert to gains
  if 'Ki' not in kwargs.keys():
    if 'Ti' in kwargs.keys():
      Ki = 1./kwargs['Ti']
    else:
      Ki = 0
  else:
    Ki = kwargs['Ki']
  if 'Kd' not in kwargs.keys():
    if 'Td' in kwargs.keys():
      Kd = 1./kwargs['Td']
    else:
      Kd = 0
  else:
    Kd = kwargs['Kd']

  return Ki, Kd


def pidstd(Kp, N=100, **kwargs):
  """
  creates the pid controller in the ideal/standard format:

  C = Kp*(1 + (1/Ti)*(1/s) + Td*s/((Td/N)s + 1))
  """
  # get inputs
  Ki, Kd = _handle_pid_inputs(**kwargs)

  s = cm.tf('s')  # make building block

  # build the equation
  Hpid = Kp*(1. + Ki*(1./s) + Kd*(N/(1. + N*(1./s))))

  return Hpid


def pidpar(Kp, **kwargs):
  """
  define a PID controller in `parallel` format
  """
  Ki, Kd = _handle_pid_inputs(**kwargs)
  if 'Tf' not in kwargs.keys():
    Tf = np.inf
  else:
    Tf = kwargs['Tf']

  s = cm.tf('s')
  Hpid = Kp + Ki/s + Kd*s/(Tf*s + 1.)

  return Hpid


def pid(Kp, N=100, form='ideal', force_statespace=True, dt=0., **kwargs):
  """
  PID controller according to the ideal scheme:
  H(s) = P(1 + I/s + D(N/(1 + N/s)))

  Note that the derivative portion is implemented as a fedback integrator

  Note that N = 1/Tf
  forms: series, parallel, standard
  """
  if form.startswith("standard") or form.startswith("ideal"):
    Hpid = pidstd(Kp, N=N, **kwargs)
  elif form.startswith("parallel"):
    Hpid = pidpar(Kp, **kwargs)
  else:
    raise NotImplementedError("Other forms than `standard`, and `parallel` are not" +
                              " implemented yet")

  Hpid.dt = dt
  if force_statespace:
    Hpid = cm.tf2ss(Hpid)

  return Hpid


def f2w(freq):
  """
  convert frequencies to angular frequencies (omega == w)
  """
  return 2*np.pi*freq


def w2f(omega):
  """
  convert angular frequencies in frequencies
  """
  return omega/(2*np.pi)


def split_s_plane_coords(s_coords, Hz=False):
  """
  split any s plane coordinates into dcay and frequency
  """
  if Hz:
    sfy = 1./(2*np.pi)
  else:
    sfy = 1.

  return aux.split_complex(s_coords, sfx=1, sfy=sfy)
