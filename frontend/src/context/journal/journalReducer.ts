import { Journal, JournalState } from '../../types/Journal';

type JournalAction =
  | { type: 'GET_JOURNALS'; payload: Journal[] }
  | { type: 'GET_CURRENT'; payload: Journal }
  | { type: 'CLEAR_CURRENT' }
  | { type: 'ADD_JOURNAL'; payload: Journal }
  | { type: 'UPDATE_JOURNAL'; payload: Journal }
  | { type: 'DELETE_JOURNAL'; payload: string }
  | { type: 'FILTER_JOURNALS'; payload: string }
  | { type: 'CLEAR_FILTER' }
  | { type: 'JOURNAL_ERROR'; payload: string }
  | { type: 'CLEAR_JOURNALS' };

const journalReducer = (state: JournalState, action: JournalAction): JournalState => {
  switch (action.type) {
    case 'GET_JOURNALS':
      return {
        ...state,
        journals: action.payload,
        loading: false,
      };
    case 'GET_CURRENT':
      return {
        ...state,
        current: action.payload,
        loading: false,
      };
    case 'CLEAR_CURRENT':
      return {
        ...state,
        current: null,
      };
    case 'ADD_JOURNAL':
      return {
        ...state,
        journals: [action.payload, ...state.journals],
        loading: false,
      };
    case 'UPDATE_JOURNAL':
      return {
        ...state,
        journals: state.journals.map((journal) =>
          journal._id === action.payload._id ? action.payload : journal
        ),
        loading: false,
      };
    case 'DELETE_JOURNAL':
      return {
        ...state,
        journals: state.journals.filter((journal) => journal._id !== action.payload),
        loading: false,
      };
    case 'FILTER_JOURNALS':
      return {
        ...state,
        filtered: state.journals.filter((journal) => {
          const regex = new RegExp(`${action.payload}`, 'gi');
          return (
            journal.instrument.match(regex) ||
            journal.setup.match(regex) ||
            journal.notes?.match(regex)
          );
        }),
      };
    case 'CLEAR_FILTER':
      return {
        ...state,
        filtered: null,
      };
    case 'JOURNAL_ERROR':
      return {
        ...state,
        error: action.payload,
        loading: false,
      };
    case 'CLEAR_JOURNALS':
      return {
        ...state,
        journals: [],
        filtered: null,
        error: null,
        current: null,
      };
    default:
      return state;
  }
};

export default journalReducer;