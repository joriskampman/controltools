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
import pdb  # noqa
from scipy.linalg import block_diag


def merge_uncoupled_ss(Hlist, squeeze_inputs=True):
  """
  merge UNCOUPLED state space objects, this is handy to create a major single state space object
  from parts

  arguments:
  ----------
  Hlist : list of control objects
          Maybe state-space representationso or transfer functions, may be siso or mimo too
  squeeze_inputs : bool, default=True
                   Whether to check if there are multiple inputs which are equal. If set to False,
                   the number of inputs will equal the number of outputs
  
  returns:
  --------
  Hmerged : control state space representation
            The merged state space representation
  """

  # merge to single state space representation
  As = []
  Bs = []
  Cs = []
  Ds = []
  for H_ in Hlist:
    A, B, C, D = cm.ssdata(H_)
    As.append(A)
    Bs.append(B)
    Cs.append(C)
    Ds.append(D)

  # merge state space matrices into bigger block (no coupling)
  A = block_diag(*As)
  B = block_diag(*Bs)
  C = block_diag(*Cs)
  D = block_diag(*Ds)

  if squeeze_inputs:
    # remove common inputs
    tf_D = np.isclose(np.sum(np.abs(D), axis=0), 0.)
    tf_B = np.isclose(np.sum(np.abs(B), axis=0), 0.)
    tf_keep = ~(tf_D*tf_B)

    B = B[:, tf_keep]
    D = D[:, tf_keep]

  # make into merged state space
  Hmerged = cm.ss(A, B, C, D)

  return Hmerged


def load_models(files, dirname='', states_to_ignore=[], vwinds_wanted=None, azis_wanted=None):
  """
  get a list of plant linearizations
  """
  files = aux.listify(files)

  Hplants_list = []
  for file in files:
    Hplants, vwinds, azis = load_state_space_model_file(file, dirname=dirname,
                                                        states_to_ignore=states_to_ignore)

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

        Hplants_list.append(Hplants[iwind, iazi])

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
    iin = aux.substr2index(iin, Hplant.inputnames)

  if isinstance(iout, str):
    iout = aux.substr2index(iout, Hplant.outputnames)

  return iin, iout


def plot_bode(Hplant, iin=0, iout=0, omega_limits=[1e-5, 1e2], omega_num=1e4, dB=True, Hz=True,
              show_margins=False, show_nyquist=False, fig=None, axs=None, color='b',
              linestyle='-', show_legend=True, label=None):
  """
  plot a single bode plot
  """
  iin, iout = _handle_iin_iout(iin, iout, Hplant)

  # plot all possible transfer functions
  if fig is None:
    fig = plt.figure(aux.figname("stability plots"))
    if show_nyquist:
      gs = fig.add_gridspec(2, 2)
    else:
      gs = fig.add_gridspec(2, 1)
    axs = [fig.add_subplot(gs[0, 0])]
    axs.append(fig.add_subplot(gs[1, 0], sharex=axs[0]))
    fig.suptitle("Stability plots", fontweight="bold", fontsize=12)
    axs[0].set_title("Bode: Magnitude")
    axs[0].set_xlabel("Frequency [Hz]")
    axs[0].set_ylabel("magnitude [dB]")
    axs[0].grid(True)
    axs[0].axhline(0, color='k', linestyle=':')
    axs[1].set_title("Bode: Phase")
    axs[1].set_xlabel("Frequency [Hz]")
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

  mags, phs, omegas = cm.bode_plot(Hsiso_this, dB=True, Hz=True, Plot=False,
                                   omega_limits=omega_limits, omega_num=omega_num)

  # convert phs to interval -inf, 0 to be able to verify the phase margins correctly
  if label is None:
    label = "{} -> {}".format(namein, nameout)

  axs[0].semilogx(omegas/(2*np.pi), cm.mag2db(mags), "-", label=label, color=color,
                  linestyle=linestyle)
  axs[1].semilogx(omegas/(2*np.pi), np.rad2deg(phs), "-", label=label,
                  color=color, linestyle=linestyle)

  gm_, pm_, sm_, wg_, wp_, ws_ = cm.stability_margins(Hsiso_this, returnall=True)

  for wg__, gm__ in zip(wg_, gm_):
    gmdb = cm.mag2db(gm__)
    wghz = wg__/(2*np.pi)
    axs[0].plot(wghz*np.array([1., 1.]), [0., -gmdb], ':', color=color)
    axs[0].text(wghz, -gmdb/2, "{:0.0f}".format(-gmdb), ha='center', va='center',
                fontsize=7, fontweight='bold', color=color, backgroundcolor='w',
                bbox={'pad': 0.1, 'color': 'w'})

  for wp__, pm__ in zip(wp_, pm_):
    # check if it is relative to -180 or +180
    iomega = aux.get_closest_index(wp__, omegas)
    nof_folds = np.abs(np.fix(phs[iomega]/(2*np.pi)))

    offset = -(nof_folds*2. + 1.)*180.

    wphz = wp__/(2*np.pi)
    axs[1].plot(wphz*np.array([1., 1.]), [offset + pm__, -180], ':', color=color)
    axs[1].text(wphz, (offset + pm__)/2 - 90., "{:0.0f}".format(pm__),
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
    axs[0].legend(fontsize=8, loc='upper right')

  plt.show(block=False)

  return fig, axs


def plot_single_plant_bodes(Hplant, iins=None, iouts=None, omega_limits=[1e-3, 1e3],
                            omega_num=1e5, dB=True, Hz=True, show_margins=False,
                            show_nyquist=False, fig=None, axs=None, color=None,
                            show_legend=True):
  """
  generate all bode plots for a plant
  """

  # handle input_indices and output_indices
  if iins is None:
    iins = np.r_[0:Hplant.inputs]
  else:
    iins = aux.arrayify(iins)
    # if isinstance(iins, (list, tuple)):
    #   iins = np.array(iins)
    # elif isinstance(iins, (int)):
    #   iins = np.array([iins,], dtype=np.int)

  if iouts is None:
    iouts = np.r_[0:Hplant.outputs]
  else:
    iouts = aux.arrayify(iouts)
    # if isinstance(iouts, (list, tuple)):
    #   iouts = np.array(iouts)
    # elif isinstance(iouts, (int)):
    #   iouts = np.array([iouts,], dtype=np.int)

  # determine the colors
  nof_bodes = iins.size*iouts.size
  if color is None:
    colors = aux.jetmod(nof_bodes, 'vector', bright=True)
  else:
    colors = color.reshape(-1, 3)

  iplot = -1
  for iin in iins:
    for iout in iouts:
      iplot += 1

      (fig, axs) = plot_bode(Hplant, iin, iout, omega_limits=omega_limits,
                             omega_num=omega_num, dB=dB, Hz=Hz, show_margins=show_margins,
                             show_nyquist=show_nyquist, fig=fig, axs=axs,
                             color=colors[iplot, :], linestyle='-', show_legend=show_legend)

  return fig, axs


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


def _remove_states_single(Hplant, states_to_ignore):
  """
  remove the states of a single plant
  """
  is_valid_state = find_valid_states(Hplant.statenames, states_to_ignore)
  A = Hplant.A[is_valid_state, :]
  A = A[:, is_valid_state]
  B = Hplant.B[is_valid_state, :]
  C = Hplant.C[:, is_valid_state]

  Hplant.states = is_valid_state.sum()
  Hplant.statenames = Hplant.statenames[is_valid_state]
  Hplant.A = A
  Hplant.B = B
  Hplant.C = C

  # nothing to return, because Hplant object is modified in place


def remove_states(Hplants, states_to_ignore):
  '''
  remove the unwanted states or single or multiple plants (i.e., a ndarray)
  '''
  if isinstance(Hplants, (list, np.ndarray)):
    for Hplant in Hplants:
      _remove_states_single(Hplant, states_to_ignore)
  else:
    _remove_states_single(Hplants, states_to_ignore)


def load_state_space_model_file(filename, dirname='.', states_to_ignore=[]):
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

  is_valid_state = find_valid_states(statenames, states_to_ignore)
  statenames = statenames[is_valid_state]

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
    As = np.swapaxis(As, 2, 3)
    Bs = np.swapaxis(Bs, 2, 3)
    Cs = np.swapaxis(Cs, 2, 3)
    Ds = np.swapaxis(Ds, 2, 3)

  for ivw, vwind in enumerate(vwinds):
    for iazi, azi in enumerate(azis):
      A = As[..., ivw, iazi]
      B = Bs[..., ivw, iazi]
      C = Cs[..., ivw, iazi]
      D = Ds[..., ivw, iazi]

      if np.prod(is_valid_state) == 0:
        A = A[is_valid_state, :]
        A = A[:, is_valid_state]
        B = B[is_valid_state, :]
        C = C[:, is_valid_state]

      Hplant = cm.ss(A, B, C, D)
      Hplant.inputnames = inputnames
      Hplant.outputnames = outputnames
      Hplant.statenames = statenames
      Hplant.wind_speed = vwind
      Hplant.azimuth = azi

      Hplant.dirname = dirname
      Hplant.filename = filename

      Hplants[ivw, iazi] = Hplant

  return Hplants, vwinds, azis


def make_siso(Hplant, iinput, ioutput):
  """
  make a MIMO system into a SISO system
  """
  iinput, ioutput = _handle_iin_iout(iinput, ioutput, Hplant)

  siso = cm.rss(states=Hplant.states, inputs=1, outputs=1)
  siso.A = Hplant.A.copy()
  siso.B = Hplant.B[:, iinput].reshape(-1, 1)
  siso.C = Hplant.C[ioutput, :].reshape(1, -1)
  siso.D = Hplant.D[ioutput, iinput].reshape(1, 1)

  if hasattr(Hplant, 'inputnames'):
    siso.inputnames = np.array([Hplant.inputnames[iinput],])
  if hasattr(Hplant, 'outputnames'):
    siso.outputnames = np.array([Hplant.outputnames[ioutput],])
  if hasattr(Hplant, 'statenames'):
    siso.statenames = Hplant.statenames

  return siso


def loadpj(items=None, dirname=None, filenames=None):
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


def notch(omega_notch, damping_factor):
  """
  returns the transfer function nof a notch filter
  """
  return cm.tf([1, 0, omega_notch**2], [1, 2*damping_factor*omega_notch, omega_notch**2])


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
  Hpid = Kp*(1. + Ki*(1./s) + Kd*(N/(1. + N*(1/s))))

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


def pid(Kp, N=100, form='ideal', **kwargs):
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

  return Hpid
