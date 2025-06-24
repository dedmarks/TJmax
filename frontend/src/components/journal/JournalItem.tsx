import React, { useContext } from 'react';
import { Link } from 'react-router-dom';
import JournalContext from '../../context/journal/JournalContext';
import { Journal } from '../../types/Journal';
import { formatDate, formatCurrency, formatNumber } from '../../utils/formatters';

interface JournalItemProps {
  journal: Journal;
}

const JournalItem: React.FC<JournalItemProps> = ({ journal }) => {
  const journalContext = useContext(JournalContext);
  const { deleteJournal, setCurrent } = journalContext;

  const {
    _id,
    instrument,
    setup,
    direction,
    entryDate,
    outcome,
    pnl,
    rewardToRisk,
  } = journal;

  const onDelete = () => {
    deleteJournal(_id);
  };

  const getOutcomeClass = (outcome: string | undefined) => {
    if (!outcome || outcome === 'Open') return '';
    return outcome === 'Win' ? 'text-success' : outcome === 'Loss' ? 'text-danger' : 'text-warning';
  };

  return (
    <tr>
      <td>{formatDate(entryDate)}</td>
      <td>{instrument}</td>
      <td>{setup}</td>
      <td className={direction === 'Long' ? 'text-success' : 'text-danger'}>{direction}</td>
      <td className={getOutcomeClass(outcome)}>{outcome || 'Open'}</td>
      <td className={pnl && pnl > 0 ? 'text-success' : pnl && pnl < 0 ? 'text-danger' : ''}>
        {pnl ? formatCurrency(pnl) : 'N/A'}
      </td>
      <td>{rewardToRisk ? formatNumber(rewardToRisk) : 'N/A'}</td>
      <td>
        <Link to={`/journal/${_id}`} className="btn btn-sm btn-info mr-1">
          <i className="fas fa-eye"></i>
        </Link>
        <Link to={`/journal/edit/${_id}`} className="btn btn-sm btn-warning mr-1">
          <i className="fas fa-edit"></i>
        </Link>
        <button onClick={onDelete} className="btn btn-sm btn-danger">
          <i className="fas fa-trash"></i>
        </button>
      </td>
    </tr>
  );
};

export default JournalItem;