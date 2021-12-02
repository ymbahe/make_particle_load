import numpy as np
import time

class TimeStamp:
    """Class for recording the runtime of program sections."""
    
    def __init__(self, verbose=False):
        """Constructor, set up (empty) list and initialize starting time."""
        self.timeList = []
        self.markList = []
        self.prevTime = time.time()

        self.otherTime = []   # Delta_t from other
        self.otherMark = []   # Mark from other
        self.otherInd = []    # Internal index of which times are part
        self.verbose = verbose

    def set_time(self, mark):                          # Class: TimeStamp
        """
        Save a 'timestamp' at a milestone in processing this snapshot.
        
        What is stored internally is the time elapsed since the last
        'stamping' call. The mark is stored in a separate list.
        
        Parameters:
        -----------
        mark : string
            A text string describing the block that has ended at the time of
            calling this function (e.g. 'Reading in particles').
        """
        self.timeList.append(time.time() - self.prevTime)
        self.markList.append(mark)
        self.prevTime = time.time()
        
    def get_time(self, mark=None):                      # Class: TimeStamp
        """
        Retrieve the length of a previously recorded time-interval.

        Parameters:
        -----------
        mark : string, optional
            The string description that was saved with the timestamp. The
            time interval of this mark is returned. If none is specified, 
            the last interval is returned.
                    
            If there are multiple instances of 'mark', the first is retrieved.
            If 'mark' does not match any of the recorded interval labels, 
            the program aborts into pdb. 

        Returns:
        --------
        delta_t : float
            The time, in seconds, of the desired time interval.
        """
        
        if not self.timeList:
            print("No time events have been recorded!")
            set_trace()

        if mark is None:
            return self.timeList[-1]
        else:
            ind_mark = np.nonzero(self.markList == mark)[0]
            if len(ind_mark) == 0:
                print("The mark '" + mark + "' was not found in the time "
                      "list. Please investigate.")
                set_trace()
            return self.timeList[ind_mark[0]]

    def add_counters(self, marks):                     # Class: TimeStamp
        """
        Add a number of (empty) time counters to the lists.

        These counters can be filled later with increase_time().

        Parameters:
        -----------
        marks : list of str
            The marks of the newly created counters.

        Returns:
        --------
        indices : list of int
            The indices of the newly created counters.
        """
        
        indices = []

        for imark in marks:
            self.timeList.append(0)
            self.markList.append(imark)
            indices.append(len(self.timeList)-1)

        return indices

    def start_time(self):                             # Class: TimeStamp
        """Re-set the internal clock, i.e. start a new time interval."""
        self.prevTime = time.time()

    def copy_times(self, other):                      # Class: TimeStamp
        """
        Copy timings from other structure, increasing existing counters.
        
        Note that, if there are multiple instances of a given mark, only
        the first one will be incremented.
  
        This function differs from import_times in that it does not create
        a new sub-level of time-counters. Also (like import_times), it 
        does not save sub-counters of the other structure.
        """

        if not other.timeList:
            if self.verbose:
                print("No time events have been recorded in other list.")
            return

        # Import time marks one-by-one:
        for iitime in range(len(other.timeList)):
            itime = other.timeList[iitime]
            imark = other.markList[iitime]
            
            # Search for already-started counters:
            ind = np.nonzero(np.array(self.markList) == imark)[0]
            if len(ind) == 0:
                self.timeList.append(itime)
                self.markList.append(imark)
            else:
                # If counter already exists, just add to it:
                self.timeList[ind[0]] += itime
        
    def import_times(self, other):                    # Class: TimeStamp
        """
        Increment internal counters with (partial) timings from other list.

        This is used to transfer timings from processing ref snaps to the 
        target snap. A separate time list is created and 'linked' 
        to the currently running time-keeping interval in this list.

        Any 'other' (sub-)lists already saved in the other list are not
        imported. If that's a problem, change the code.

        Parameters:
        -----------
        other : TimeList instance
            The list from which time stamps should be imported.
        """

        if not other.timeList:
            if self.verbose:
                print("No time events have been recorded in other list.")
            return


        # Determine internal index to which imported times are `attached':
        currIndex = len(self.timeList)

        # Import time marks one-by-one:
        for iitime in range(len(other.timeList)):
            itime = other.timeList[iitime]
            imark = other.markList[iitime]
            
            # Search for already-started counters:
        
            makeNewEntry = True
            if len(self.otherMark) > 0:
                ind = np.nonzero((np.array(self.otherMark) == imark) & 
                                 (np.array(self.otherInd) == currIndex))[0]
                if len(ind) > 0:
                    # If counter already exists, just add to it:
                    self.otherTime[ind[0]] += itime
                    makeNewEntry = False
            if makeNewEntry:
                self.otherTime.append(itime)
                self.otherMark.append(imark)
                self.otherInd.append(currIndex)

                                                       # Class: TimeStamp
    def increase_time(self, mark=None, index=None):
        """Increase an existing time counter."""
    
        if index is  None:
            if mark is not None:
                index = self.markList.index(mark)
            else:
                print("Cannot increase time without knowing where!")
                set_trace()

        self.timeList[index] += (time.time()-self.prevTime)
        self.prevTime = time.time()

                                                        # Class: TimeStamp
    def print_time_usage(self, caption=None, mode='detailed',
                         minutes=False, percent=True):          
        """
        Print a report on the internally stored times.

        Parameters:
        -----------
        caption : str, optional
            A string to display at the beginning of the report. If None,
            no caption string is printed.

        mode : str, optional
            Defines how output should be structured. Options are:
            -- 'detailed' (default):
               All sub-counters are printed separately for their
               respective top-level counter.
            -- 'top':
               Only top-level counters are printed.
            -- 'sub':
               All sub-counters with the same mark are combined and
               printed. No top-level information is printed.
        """

        if mode not in ['detailed', 'top', 'sub']:
            print("Wrong mode option '" + mode + "' for "
                  "print_time_usage(). Please investigate.")
            set_trace()
        
        if caption is None:
            caption = "Finished "

        self.minutes = minutes
        self.percent = percent

        self.fullTime = np.sum(np.array(self.timeList))

        print("")
        print("-" * 70)
        if minutes:
            print(caption + " ({:.2f} min.)" .format(self.fullTime/60))
        else:
            print(caption + " ({:.4f} sec.)" .format(self.fullTime))
        print("-" * 70)

        if mode in ['detailed', 'top']:
            markLength = len(max(self.markList, key=len))+3
            if self.otherMark:
                otherMarkLength = len(max(self.otherMark, key=len))+3+5
                if otherMarkLength < markLength+3:
                    otherMarkLength = markLength+3

            for iimark, imark in enumerate(self.markList):
                print((imark+':').ljust(markLength) + 
                      self._tstr(self.timeList[iimark]))
                
                if mode == 'detailed':
                    ind_other = np.nonzero(
                        np.array(self.otherInd) == iimark)[0]            
                    for iiother in ind_other:
                        print(("  -- "+self.otherMark[iiother]+':').ljust(
                            otherMarkLength) + 
                              self._tstr(self.otherTime[iiother]))
        else:
            # Print combine sub-counters
            sub_unique = set(self.otherMark)
            markLength = len(max(sub_unique, key=len))+3
            for iisub, isub in enumerate(sub_unique):
                ind_this = np.nonzero(np.array(self.otherMark) == isub)[0]
                if len(ind_this) == 0: set_trace()

                sumTime = np.sum(np.array(self.otherTime)[ind_this])
                print((isub+':').ljust(markLength) + self._tstr(sumTime))
            
        print("-" * 70)
        print("")

    def _tstr(self, time):
        """Produce formatted time string"""
        
        tstr = "{:5.2f} sec." .format(time)
        if not self.minutes and not self.percent:
            return tstr

        if self.minutes and self.percent:
            return (tstr + " ({:.2f} min., {:.1f}%)" 
                    .format(time/60, time/self.fullTime*100))

        if self.minutes:
            return tstr + " ({:.2f} min.)" .format(time/60)

        if self.percent:
            return tstr + " ({:.1f}%)" .format(time/self.fullTime*100)
