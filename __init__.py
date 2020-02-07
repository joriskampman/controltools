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
from warnings import warn
import matplotlib.pyplot as plt
import pdb


def load_multiple_models(files, dirname='', states_to_ignore=[], vwinds_wanted=None,
                         azis_wanted=None):
  """
  get a list of plant linearizations
  """
  if isinstance(files, str):
    files = [files,]

  Hplants_list = []
  for file in files:
    Hplants, vwinds, azis = load_state_space_model(file, dirname=dirname,
                                                   states_to_ignore=states_to_ignore)

    if vwinds_wanted is None:
      vwinds_wanted = vwinds.copy()

    if azis_wanted is None:
      azis_wanted = azis.copy()

    for vwind_wanted in vwinds_wanted:
      iwind = get_closest_index(vwinds, vwind_wanted)
      for azi_wanted in azis_wanted:
        iazi = get_closest_index(azis, azi_wanted)

        Hplants_list.append(Hplants[iwind, iazi])

  return Hplants_list


def plot_multiple_bodes(Hplants_list, inputname, outputname, split=False, show_nyquist=False):
  """
  plot multiple siso bodes
  """
  fig = None
  axs = None
  colors = aux.jetmod(len(Hplants_list), 'vector', bright=True)
  for color, Hplant in zip(colors, Hplants_list):
    iin = substr2index(inputname, Hplant.inputnames)
    iout = substr2index(outputname, Hplant.outputnames)
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


def plot_bode(Hplant, iin, iout, omega_limits=[1e-3, 1e3], omega_num=1e5, dB=True, Hz=True,
              show_margins=False, show_nyquist=False, fig=None, axs=None, color='b',
              linestyle='-', show_legend=True, label=None):
  """
  plot a single bode plot
  """
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
                                   omega_limits=[1e-4, 1e2], omega_num=1e4)

  # # bring phases to between -180 and 180
  # phs = np.angle(np.exp(1j*phs))
  # # bring phases between -360 and 0
  # iabove0 = np.nonzero(phs > 0.)
  # phs[iabove0] -= 2*np.pi

  lines = []
  texts = []
  # convert phs to interval -inf, 0 to be able to verify the phase margins correctly
  if label is None:
    label = "{} -> {}".format(namein, nameout)
  line_ = axs[0].semilogx(omegas/(2*np.pi), cm.mag2db(mags), "-", label=label, color=color,
                          linestyle=linestyle)
  lines.append(line_)
  line_ = axs[1].semilogx(omegas/(2*np.pi), np.rad2deg(phs), "-", label=label,
                          color=color, linestyle=linestyle)
  lines.append(line_)

  gm_, pm_, sm_, wg_, wp_, ws_ = cm.stability_margins(Hsiso_this, returnall=True)

  for wg__, gm__ in zip(wg_, gm_):
    gmdb = cm.mag2db(gm__)
    wghz = wg__/(2*np.pi)
    line_ = axs[0].plot(wghz*np.array([1., 1.]), [0., -gmdb], ':', color=color)
    lines.append(line_)
    text_ = axs[0].text(wghz, -gmdb/2, "{:0.0f}".format(-gmdb), ha='center', va='center',
                        fontsize=7, fontweight='bold', color=color, backgroundcolor='w',
                        bbox={'pad': 0.1, 'color': 'w'})
    texts.append(text_)

  for wp__, pm__ in zip(wp_, pm_):
    # check if it is relative to -180 or +180
    iomega = get_closest_index(omegas, wp__)
    nof_folds = np.abs(np.fix(phs[iomega]/(2*np.pi)))

    offset = -(nof_folds*2. + 1.)*180.

    wphz = wp__/(2*np.pi)
    line_ = axs[1].plot(wphz*np.array([1., 1.]), [offset + pm__, -180], ':',
                        color=color)
    text_ = axs[1].text(wphz, (offset + pm__)/2 - 90., "{:0.0f}".format(pm__),
                        ha='center', va='center', fontsize=7, fontweight='bold', color=color,
                        bbox={'pad': 0.1, 'color': 'w'})

    lines.append(line_)
    texts.append(text_)

  if show_nyquist:
    nqi, nqq, nqf = cm.nyquist_plot(Hsiso_this, omega=omegas, Plot=False)
    cvec = aux.color_vector(omegas.size, color, os=0.25)
    line_ = axs[2].scatter(nqi, nqq, s=2, c=cvec)
    lines.append(line_)
    nqii = np.interp(ws_, nqf, nqi)
    nqqi = np.interp(ws_, nqf, nqq)
    for ipt in range(nqii.size):
      line_ = axs[2].plot([-1., nqii[ipt]], [0., nqqi[ipt]], '-', color=color)
      lines.append(line_)

  if show_legend:
    axs[0].legend(fontsize=8, loc='upper right')

  plt.show(block=False)

  return fig, axs, lines, texts


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
    if isinstance(iins, (list, tuple)):
      iins = np.array(iins)
    elif isinstance(iins, (int)):
      iins = np.array([iins,], dtype=np.int)

  if iouts is None:
    iouts = np.r_[0:Hplant.outputs]
  else:
    if isinstance(iouts, (list, tuple)):
      iouts = np.array(iouts)
    elif isinstance(iouts, (int)):
      iouts = np.array([iouts,], dtype=np.int)

  # determine the colors
  nof_bodes = iins.size*iouts.size
  if color is None:
    colors = aux.jetmod(nof_bodes, 'vector')
  else:
    colors = color.reshape(-1, 3)

  lines = []
  texts = []
  iplot = -1
  for iin in iins:
    for iout in iouts:
      iplot += 1

      (fig,
       axs,
       lines_,
       texts_) = plot_bode(Hplant, iin, iout, omega_limits=omega_limits,
                           omega_num=omega_num, dB=dB, Hz=Hz, show_margins=show_margins,
                           show_nyquist=show_nyquist, fig=fig, axs=axs,
                           color=colors[iplot, :], linestyle='-', show_legend=show_legend)

      lines.append(lines_)
      texts.append(texts_)

  return fig, axs, lines, texts


def get_closest_index(values, value_wanted, suppress_warnings=False):
  """
  get the closest index
  """
  ifnd_arr = np.argwhere(np.isclose(values, value_wanted)).ravel()
  if ifnd_arr.size > 0:
    ifnd = ifnd_arr.item()
  else:
    # get the closest
    ifnd = np.argmin(np.abs(values - value_wanted)).ravel()[0]
    if not suppress_warnings:
      warn("There is no `exact` match for value = {}. Taking the closest value = {}".
           format(value_wanted, values[ifnd]))

  return ifnd


def substr2index(substring, strlist):
  """
  find the indices of a certain substring
  """
  index = np.argwhere([substring.lower() in elm.lower() for elm in strlist]).item()

  return index


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


def load_state_space_model(filename, dirname='.', states_to_ignore=[]):
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


def make_siso(plant, iinput, ioutput):
  """
  make a MIMO system into a SISO system
  """

  siso = cm.rss(states=plant.states, inputs=1, outputs=1)
  siso.A = plant.A.copy()
  siso.B = plant.B[:, iinput].reshape(-1, 1)
  siso.C = plant.C[ioutput, :].reshape(1, -1)
  siso.D = plant.D[ioutput, iinput].reshape(1, 1)

  if hasattr(plant, 'inputnames'):
    siso.inputnames = np.array([plant.inputnames[iinput],])
  if hasattr(plant, 'outputnames'):
    siso.outputnames = np.array([plant.outputnames[ioutput],])
  if hasattr(plant, 'statenames'):
    siso.statenames = plant.statenames

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


def pidstd(Kp, Ti=0, Td=0, N=100):
  """
  creates the pid controller in the ideal/standard format:

  C = Kp*(1 + (1/Ti)*(1/s) + Td*s/((Td/N)s + 1))
  """
  Hp = cm.tf(1)
  Hi = cm.tf(1, [Ti, 0])
  Hd = cm.tf([Td, 0], [Td/N, 1])

  Hpid = Kp*(Hp + Hi + Hd)

  return Hpid


def pidpar(Kp, Ki=0, Kd=0, Tf=0):
  """
  define a PID controller in `parallel` format
  """
  Hpid = Kp* + cm.tf(Ki, [1, 0]) + cm.tf([Kd, 0], [Tf, 1])

  return Hpid


def pid(Kp, Ki=0, Kd=0, Ti=0, Td=0, Tf=0, N=100, form='ideal'):
  """
  PID controller according to the ideal scheme:
  H(s) = P(1 + I/s + D(N/(1 + N/s)))

  Note that the derivative portion is implemented as a fedback integrator

  Note that N = 1/Tf
  forms: series, parallel, standard
  """
  if form.startswith("standard") or form.startswith("ideal"):
    Hpid = pidstd(Kp, Ti=Ti, Td=Td, N=N)
  elif form.startswith("parallel"):
    Hpid = pidpar(Kp, Ki=Ki, Kd=Kd, Tf=Tf)
  else:
    raise NotImplementedError("Other forms than `standard`, and `parallel` are not" +
                              " implemented yet")

  return Hpid
