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

  # check if last req_buffer smaples are within perc
  if np.max(np.abs(ys[-nbuffer:] - yfinal)) > perc/100:
    warn('Shit has not settled upto {:d} samples before the end.'.format(nbuffer))
    return np.nan

  yrefinv = ys[-1::-1]  # from right to left
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

  return t_settle


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


def _prune(Hsys, strlist, inout):
  """
  prune the inputs of a system
  """
  if inout == 'in':
    allnames = Hsys.inputnames.copy()
  elif inout == 'out':
    allnames = Hsys.outputnames.copy()

  indices2keep = []

  nameslist = []
  for i2k in strlist:
    ifnd = aux.find_elm_containing_substrs(i2k, allnames, nreq=1, strmatch="all")
    indices2keep.append(ifnd)
    nameslist.append(allnames[ifnd])

  # set all invalids to zero
  if inout == 'in':
    Hsys.B = Hsys.B[:, indices2keep]
    Hsys.D = Hsys.D[:, indices2keep]
    Hsys.inputs = len(indices2keep)
    Hsys.inputnames = nameslist
  elif inout == 'out':
    Hsys.C = Hsys.C[indices2keep, :]
    Hsys.D = Hsys.D[indices2keep, :]
    Hsys.outputs = len(indices2keep)
    Hsys.outputnames = nameslist

  return indices2keep


def prune_ios(Hsys, inputnames=None, outputnames=None):
  """
  prune the inputs and outputs from a system
  """
  iins = np.r_[:Hsys.inputs]
  iouts = np.r_[:Hsys.outputs]
  if inputnames is not None:
    iins = _prune(Hsys, inputnames, 'in')

  if outputnames is not None:
    iouts = _prune(Hsys, outputnames, 'out')

  return iins, iouts


def build_system(blocks, Qstrings, tag="system", opens=None, shorts=None):
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
  Qstrings : array-like of tuples of strings
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

  # next step: define interconnections based on input and outputnames
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
    sys_inputnames += block.inputnames
    sys_outputnames += block.outputnames
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

  return Hsys


def replace_io_strings(nameslist, replacements, show_warnings=False):
  """
  modifiy entries in the list by the replacements
  """
  outlist = nameslist.copy()
  for repl in replacements:
    try:
      ifnd = aux.find_elm_containing_substrs(repl[0], nameslist, nreq=1, strmatch="all")
      outlist[ifnd] = repl[1]
    except Exception:
      if show_warnings:
        warn("nothing found for `{}` in {}".format(repl[0], nameslist))

  return outlist


def make_block(btype, tag, dt=0., inames=None, onames=None, ss_or_tf='ss', keep_names=False,
               **blargs):
  """
  to be filled in
  """
  # take blargs as keyword arguments for the subfunctions
  is_plant = False
  if btype == 'plant':
    if 'ss' in blargs.keys():
      block = blargs['ss']
    elif 'ssdata' in blargs.keys():
      block = cm.StateSpace(**blargs, remove_useless=False)
    elif 'tf' in blargs.keys():
      block = cm.tf2ss(blargs['tf'])
    is_plant = True
    if inames is None:
      if hasattr(block, 'inputnames'):
        inames = block.inputnames.copy()
    if onames is None:
      if hasattr(block, 'outputnames'):
        onames = block.outputnames.copy()
  if btype == 'g':
    block = cm.StateSpace([], [], [], blargs['gain'], remove_useless=False)
  elif btype == 'short':
    block = cm.StateSpace([], [], [], 1., remove_useless=False)
  elif btype == 'open':
    block = cm.StateSpace([], [], [], 0., remove_useless=False)
  elif btype == 'inv':
    block = cm.StateSpace([], [], [], -1., remove_useless=False)
  elif btype == 'ss':
    block = blargs['ss']
  elif btype == 'ssdata':
    block = cm.StateSpace(**blargs, remove_useless=False)
  elif btype == 'tf':
    block = cm.tf2ss(blargs['tf'])
  elif btype in ['p', 'i', 'd', 'pi', 'pd', 'pid']:
    block = pid(**blargs)
  elif btype == 'nf':
    block = notch(**blargs)
  elif btype == 'bp':
    block = bandpass(**blargs)

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
    iin = aux.find_elm_containing_substrs(tuple(iin.split()), Hplant.inputnames, nreq=1)

  if isinstance(iout, str):
    iout = aux.find_elm_containing_substrs(tuple(iout.split()), Hplant.outputnames, nreq=1)

  return iin, iout


def plot_bode(Hplant, iin=0, iout=0, omega_limits=[1e-3, 1e2], omega_num=1e4, dB=True, Hz=True,
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
    if Hz:
      axs[0].set_xlabel("Frequency [Hz]")
    else:
      axs[0].set_xlabel("Angular frequency [rad/s]")
    axs[0].set_ylabel("magnitude [dB]")
    axs[0].grid(True)
    axs[0].axhline(0, color='k', linestyle=':')
    axs[1].set_title("Bode: Phase")
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
                                   omega_limits=omega_limits, omega_num=omega_num)

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


def plot_single_plant_bodes(Hplant, iins=None, iouts=None, omega_limits=[1e-3, 1e2],
                            omega_num=1e5, dB=True, Hz=True, show_margins=False,
                            show_nyquist=False, fig=None, axs=None, colors=None,
                            show_legend=True):
  """
  generate all bode plots for a plant
  """

  # handle input_indices and output_indices
  if iins is None:
    iins = np.r_[0:Hplant.inputs]
  else:
    iins = aux.arrayify(iins)

  if iouts is None:
    iouts = np.r_[0:Hplant.outputs]
  else:
    iouts = aux.arrayify(iouts)

  # determine the colors
  nof_bodes = iins.size*iouts.size
  if colors is None:
    colors = aux.jetmod(nof_bodes, 'vector', bright=True)

  iplot = -1
  for iin in iins:
    for iout in iouts:
      iplot += 1
      (fig, axs) = plot_bode(Hplant, iin, iout, omega_limits=omega_limits,
                             omega_num=omega_num, dB=dB, Hz=Hz, show_margins=show_margins,
                             show_nyquist=show_nyquist, fig=fig, axs=axs,
                             color=colors[iplot], linestyle='-', show_legend=show_legend)

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


def load_state_space_model_file(filename, dirname='.', states_to_ignore=[], dt=0):
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

      Hplant = cm.StateSpace(A, B, C, D, dt, remove_useless=False)
      # Hplant.A = A.copy()
      # Hplant.states = A.shape[0]
      # Hplant.B = B.copy()
      # Hplant.C = C.copy()
      # Hplant.D = D.copy()
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


def notch(f0, damping_ratio, gain=1., fz=None, dt=0, ss_or_tf='ss'):
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

  if ss_or_tf == 'ss':
    out = cm.tf2ss(out)
  elif ss_or_tf == 'tf':
    pass
  else:
    raise ValueError("The given value for `ss_or_tf` is not valid ({})".format(ss_or_tf))

  return out


def bandpass(f0, damping_ratio, gain=1., dt=0, ss_or_tf='ss'):
  """
  create a bandpass filter
  """
  omega0 = f2w(f0)
  num = [2*damping_ratio*omega0, 0]
  den = [1., 2*damping_ratio*omega0, omega0**2]

  out = gain*cm.tf(num, den, dt)

  if ss_or_tf == 'ss':
    out = cm.tf2ss(out)
  elif ss_or_tf == 'tf':
    pass
  else:
    raise ValueError("The given value for `ss_or_tf` is not valid ({})".format(ss_or_tf))

  return out


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


def pid(Kp, N=100, form='ideal', ss_or_tf='ss', dt=0., **kwargs):
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
  if ss_or_tf == 'ss':
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
