import React, { useReducer } from 'react';
import JournalContext from './JournalContext';
import journalReducer from './journalReducer';
import { getJournalEntries, getJournalEntryById, createJournalEntry, updateJournalEntry, deleteJournalEntry } from '../../services/journalService';
import { Journal, JournalFormData, JournalState as IJournalState } from '../../types/Journal';

interface Props {
  children: React.ReactNode;
}

const JournalState: React.FC<Props> = ({ children }) => {
  const initialState: IJournalState = {
    journals: [],
    currentJournal: null,  // Add this line
    current: null,
    filtered: null,
    loading: true,
    error: null,
  };

  const [state, dispatch] = useReducer(journalReducer, initialState);

  // Get all journal entries
  const getJournals = async (): Promise<void> => {
    try {
      const res = await getJournalEntries();

      dispatch({
        type: 'GET_JOURNALS',
        payload: res,
      });
    } catch (err: any) {
      dispatch({
        type: 'JOURNAL_ERROR',
        payload: err.response?.data.msg || 'Failed to fetch journal entries',
      });
    }
  };

  // Get journal entry by ID
  const getCurrent = async (id: string): Promise<void> => {
    try {
      const res = await getJournalEntryById(id);

      dispatch({
        type: 'GET_CURRENT',
        payload: res,
      });
    } catch (err: any) {
      dispatch({
        type: 'JOURNAL_ERROR',
        payload: err.response?.data.msg || 'Failed to fetch journal entry',
      });
    }
  };

  // Clear current journal entry
  const clearCurrent = (): void => {
    dispatch({ type: 'CLEAR_CURRENT' });
  };

  // Add journal entry
  const addJournal = async (journal: JournalFormData): Promise<void> => {
    try {
      const res = await createJournalEntry(journal);

      dispatch({
        type: 'ADD_JOURNAL',
        payload: res,
      });
    } catch (err: any) {
      dispatch({
        type: 'JOURNAL_ERROR',
        payload: err.response?.data.msg || 'Failed to add journal entry',
      });
    }
  };

  // Update journal entry
  const updateJournal = async (id: string, journal: JournalFormData): Promise<void> => {
    try {
      const res = await updateJournalEntry(id, journal);

      dispatch({
        type: 'UPDATE_JOURNAL',
        payload: res,
      });
    } catch (err: any) {
      dispatch({
        type: 'JOURNAL_ERROR',
        payload: err.response?.data.msg || 'Failed to update journal entry',
      });
    }
  };

  // Delete journal entry
  const deleteJournal = async (id: string): Promise<void> => {
    try {
      await deleteJournalEntry(id);

      dispatch({
        type: 'DELETE_JOURNAL',
        payload: id,
      });
    } catch (err: any) {
      dispatch({
        type: 'JOURNAL_ERROR',
        payload: err.response?.data.msg || 'Failed to delete journal entry',
      });
    }
  };

  // Filter journal entries
  const filterJournals = (text: string): void => {
    dispatch({
      type: 'FILTER_JOURNALS',
      payload: text,
    });
  };

  // Clear filter
  const clearFilter = (): void => {
    dispatch({ type: 'CLEAR_FILTER' });
  };

  // Clear journals
  const clearJournals = (): void => {
    dispatch({ type: 'CLEAR_JOURNALS' });
  };

  return (
    <JournalContext.Provider
      value={{
        journals: state.journals,
        currentJournal: state.currentJournal,  // Add this line
        current: state.current,
        filtered: state.filtered,
        loading: state.loading,
        error: state.error,
        getJournals,
        getCurrent,
        clearCurrent,
        addJournal,
        updateJournal,
        deleteJournal,
        filterJournals,
        clearFilter,
        clearJournals,
      }}
    >
      {children}
    </JournalContext.Provider>
  );
};

export default JournalState;