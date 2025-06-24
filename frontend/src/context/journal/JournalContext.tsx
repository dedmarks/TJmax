import { createContext } from 'react';
import { Journal, JournalFormData, JournalState } from '../../types/Journal';

interface JournalContextInterface extends JournalState {
  getJournals: () => Promise<void>;
  getCurrent: (id: string) => Promise<void>;
  clearCurrent: () => void;
  addJournal: (journal: JournalFormData) => Promise<void>;
  updateJournal: (id: string, journal: JournalFormData) => Promise<void>;
  deleteJournal: (id: string) => Promise<void>;
  filterJournals: (text: string) => void;
  clearFilter: () => void;
  clearJournals: () => void;
  setCurrent?: (journal: Journal) => void; // Add this if needed
}

const JournalContext = createContext<JournalContextInterface>({
  journals: [],
  current: null,
  currentJournal: null,
  filtered: null,
  loading: true,
  error: null,
  getJournals: async () => {},
  getCurrent: async () => {},
  clearCurrent: () => {},
  addJournal: async () => {},
  updateJournal: async () => {},
  deleteJournal: async () => {},
  filterJournals: () => {},
  clearFilter: () => {},
  clearJournals: () => {},
  setCurrent: () => {}
});

export default JournalContext;