/**
 * FactPulse Background Service Worker
 * 
 * Gère:
 * - Communication entre popup et content scripts
 * - Badge de l'extension
 * - Storage des statistiques
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

const API_URL = 'http://localhost:8000';

// ============================================================================
// MESSAGE HANDLING
// ============================================================================

/**
 * Écoute les messages des content scripts et popup.
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  
  if (message.action === 'updateBadge') {
    updateBadge(sender.tab?.id, message.status, message.text);
    sendResponse({ success: true });
  }
  
  else if (message.action === 'getStatus') {
    checkHealth().then(status => sendResponse(status));
    return true; // Keep channel open for async response
  }
  
  else if (message.action === 'saveStats') {
    saveStats(message.stats).then(() => sendResponse({ success: true }));
    return true;
  }
  
  else if (message.action === 'analysisComplete') {
    handleAnalysisComplete(sender.tab?.id, message.results);
    sendResponse({ success: true });
  }
  
  return false;
});

// ============================================================================
// BADGE MANAGEMENT
// ============================================================================

/**
 * Met à jour le badge de l'extension pour un onglet.
 */
function updateBadge(tabId, status, text = '') {
  const colors = {
    idle: '#718096',
    checking: '#ed8936',
    success: '#38a169',
    warning: '#e53e3e',
    error: '#742a2a'
  };
  
  const badgeText = {
    idle: '',
    checking: '...',
    success: '✓',
    warning: '!',
    error: '✗'
  };
  
  const color = colors[status] || colors.idle;
  const badge = text || badgeText[status] || '';
  
  if (tabId) {
    chrome.action.setBadgeBackgroundColor({ tabId, color });
    chrome.action.setBadgeText({ tabId, text: badge });
  } else {
    chrome.action.setBadgeBackgroundColor({ color });
    chrome.action.setBadgeText({ text: badge });
  }
}

// ============================================================================
// API HEALTH CHECK
// ============================================================================

/**
 * Vérifie l'état du serveur.
 */
async function checkHealth() {
  try {
    const response = await fetch(`${API_URL}/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' }
    });
    
    if (response.ok) {
      const data = await response.json();
      return {
        connected: true,
        gpu: data.gpu_available,
        models: data.models_loaded
      };
    }
  } catch (error) {
    console.error('[FactPulse] Health check failed:', error);
  }
  
  return { connected: false };
}

// ============================================================================
// STATISTICS
// ============================================================================

/**
 * Sauvegarde les statistiques.
 */
async function saveStats(newStats) {
  try {
    const stored = await chrome.storage.local.get(['stats']);
    const stats = stored.stats || { totalClaims: 0, totalTime: 0, count: 0 };
    
    stats.totalClaims += newStats.claims || 0;
    stats.totalTime += newStats.time || 0;
    stats.count += 1;
    stats.avgTime = Math.round(stats.totalTime / stats.count);
    
    await chrome.storage.local.set({ stats });
  } catch (error) {
    console.error('[FactPulse] Stats save failed:', error);
  }
}

// ============================================================================
// ANALYSIS HANDLING
// ============================================================================

/**
 * Gère la fin d'une analyse.
 */
function handleAnalysisComplete(tabId, results) {
  if (!results) return;
  
  // Déterminer le statut
  let status = 'idle';
  let badgeText = '';
  
  if (results.status === 'error') {
    status = 'error';
  } else if (results.results && results.results.length > 0) {
    const hasProblematic = results.results.some(
      r => r.verdict === 'FALSE' || r.verdict === 'PARTIALLY_TRUE'
    );
    
    if (hasProblematic) {
      status = 'warning';
      badgeText = results.results.length.toString();
    } else {
      status = 'success';
    }
  }
  
  // Mettre à jour le badge
  updateBadge(tabId, status, badgeText);
  
  // Sauvegarder les stats
  if (results.timing) {
    saveStats({
      claims: results.claims_verified || 0,
      time: results.timing.total_ms || 0
    });
  }
}

// ============================================================================
// INSTALLATION
// ============================================================================

/**
 * Premier lancement de l'extension.
 */
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    console.log('[FactPulse] Extension installed');
    
    // Initialiser le storage
    chrome.storage.local.set({
      stats: { totalClaims: 0, totalTime: 0, count: 0, avgTime: 0 },
      settings: { autoAnalyze: false, threshold: 0.7 }
    });
  }
});

// ============================================================================
// TAB UPDATES
// ============================================================================

/**
 * Réinitialise le badge quand on change de page.
 */
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete') {
    updateBadge(tabId, 'idle');
  }
});

console.log('[FactPulse] Background service worker started');
