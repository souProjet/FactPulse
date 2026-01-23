/**
 * FactPulse Content Script
 * 
 * Fonctionnalités:
 * - Extraction du texte visible de la page
 * - Communication avec le backend FastAPI
 * - Affichage des badges de statut
 * - Injection DOM minimale
 * 
 * Flux de données:
 * 1. Page chargée → extraction du texte
 * 2. Envoi au backend → attente de réponse
 * 3. Réception verdict → mise à jour badge
 * 4. Clic sur badge → affichage détails
 */

(function() {
  'use strict';

  // ============================================================================
  // CONFIGURATION
  // ============================================================================
  
  const CONFIG = {
    API_URL: 'http://localhost:8000',
    MAX_TEXT_LENGTH: 10000,
    MIN_TEXT_LENGTH: 50,
    DEBOUNCE_MS: 1000,
    TIMEOUT_MS: 5000
  };

  // ============================================================================
  // ÉTAT
  // ============================================================================
  
  let currentStatus = 'idle'; // idle, checking, success, warning, error
  let lastResults = null;
  let badgeElement = null;
  let panelElement = null;

  // ============================================================================
  // EXTRACTION DE TEXTE
  // ============================================================================
  
  /**
   * Extrait le texte visible de la page.
   * Filtre les éléments non pertinents (scripts, styles, etc.)
   */
  function extractVisibleText() {
    const ignoreTags = new Set([
      'SCRIPT', 'STYLE', 'NOSCRIPT', 'IFRAME', 'SVG', 
      'PATH', 'NAV', 'HEADER', 'FOOTER', 'ASIDE'
    ]);
    
    const ignoreClasses = new Set([
      'nav', 'menu', 'sidebar', 'footer', 'header', 
      'advertisement', 'ad', 'comment', 'comments'
    ]);
    
    function isVisible(element) {
      const style = window.getComputedStyle(element);
      return style.display !== 'none' && 
             style.visibility !== 'hidden' && 
             style.opacity !== '0';
    }
    
    function shouldIgnore(element) {
      if (ignoreTags.has(element.tagName)) return true;
      
      const classList = element.className.toLowerCase();
      for (const cls of ignoreClasses) {
        if (classList.includes(cls)) return true;
      }
      
      return false;
    }
    
    function extractText(element, texts = []) {
      if (!element || shouldIgnore(element)) return texts;
      
      // Text nodes
      for (const node of element.childNodes) {
        if (node.nodeType === Node.TEXT_NODE) {
          const text = node.textContent.trim();
          if (text.length > 10) {
            texts.push(text);
          }
        } else if (node.nodeType === Node.ELEMENT_NODE && isVisible(node)) {
          extractText(node, texts);
        }
      }
      
      return texts;
    }
    
    // Chercher le contenu principal
    const mainSelectors = [
      'article', 'main', '[role="main"]', 
      '.content', '.post', '.article',
      '#content', '#main'
    ];
    
    let mainElement = null;
    for (const selector of mainSelectors) {
      mainElement = document.querySelector(selector);
      if (mainElement) break;
    }
    
    // Fallback vers body
    const targetElement = mainElement || document.body;
    
    const texts = extractText(targetElement);
    const fullText = texts.join(' ').replace(/\s+/g, ' ').trim();
    
    // Limiter la longueur
    if (fullText.length > CONFIG.MAX_TEXT_LENGTH) {
      return fullText.substring(0, CONFIG.MAX_TEXT_LENGTH);
    }
    
    return fullText;
  }

  // ============================================================================
  // COMMUNICATION API
  // ============================================================================
  
  /**
   * Envoie le texte au backend pour analyse.
   */
  async function analyzeText(text) {
    if (text.length < CONFIG.MIN_TEXT_LENGTH) {
      return { status: 'skip', message: 'Texte trop court' };
    }
    
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), CONFIG.TIMEOUT_MS);
    
    try {
      const response = await fetch(`${CONFIG.API_URL}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text }),
        signal: controller.signal
      });
      
      clearTimeout(timeout);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      clearTimeout(timeout);
      
      if (error.name === 'AbortError') {
        return { status: 'error', message: 'Timeout' };
      }
      
      return { status: 'error', message: error.message };
    }
  }

  // ============================================================================
  // BADGE UI
  // ============================================================================
  
  /**
   * Crée ou met à jour le badge de statut.
   */
  function createBadge() {
    if (badgeElement) return badgeElement;
    
    badgeElement = document.createElement('div');
    badgeElement.id = 'factpulse-badge';
    badgeElement.innerHTML = `
      <div class="fp-badge-icon"></div>
      <div class="fp-badge-text">FactPulse</div>
    `;
    
    badgeElement.addEventListener('click', togglePanel);
    
    document.body.appendChild(badgeElement);
    return badgeElement;
  }
  
  /**
   * Met à jour le statut du badge.
   * @param {'idle'|'checking'|'success'|'warning'|'error'} status
   * @param {string} message
   */
  function updateBadge(status, message = '') {
    currentStatus = status;
    
    const badge = createBadge();
    badge.className = `fp-badge fp-badge-${status}`;
    
    const iconEl = badge.querySelector('.fp-badge-icon');
    const textEl = badge.querySelector('.fp-badge-text');
    
    const icons = {
      idle: '○',
      checking: '◐',
      success: '✓',
      warning: '⚠',
      error: '✗'
    };
    
    const texts = {
      idle: 'FactPulse',
      checking: 'Vérification...',
      success: 'Vérifié',
      warning: 'Suspect',
      error: 'Erreur'
    };
    
    iconEl.textContent = icons[status] || '○';
    textEl.textContent = message || texts[status] || 'FactPulse';
    
    // Animation pour checking
    if (status === 'checking') {
      badge.classList.add('fp-badge-pulse');
    } else {
      badge.classList.remove('fp-badge-pulse');
    }
  }

  // ============================================================================
  // PANEL UI
  // ============================================================================
  
  /**
   * Crée le panneau de détails.
   */
  function createPanel() {
    if (panelElement) return panelElement;
    
    panelElement = document.createElement('div');
    panelElement.id = 'factpulse-panel';
    panelElement.innerHTML = `
      <div class="fp-panel-header">
        <span class="fp-panel-title">FactPulse</span>
        <button class="fp-panel-close">×</button>
      </div>
      <div class="fp-panel-content">
        <div class="fp-panel-loading">Chargement...</div>
      </div>
    `;
    
    panelElement.querySelector('.fp-panel-close').addEventListener('click', hidePanel);
    
    document.body.appendChild(panelElement);
    return panelElement;
  }
  
  /**
   * Affiche/masque le panneau.
   */
  function togglePanel() {
    const panel = createPanel();
    
    if (panel.classList.contains('fp-panel-visible')) {
      hidePanel();
    } else {
      showPanel();
    }
  }
  
  function showPanel() {
    const panel = createPanel();
    panel.classList.add('fp-panel-visible');
    renderPanelContent();
  }
  
  function hidePanel() {
    if (panelElement) {
      panelElement.classList.remove('fp-panel-visible');
    }
  }
  
  /**
   * Met à jour le contenu du panneau.
   */
  function renderPanelContent() {
    if (!panelElement) return;
    
    const contentEl = panelElement.querySelector('.fp-panel-content');
    
    if (!lastResults) {
      contentEl.innerHTML = `
        <div class="fp-panel-empty">
          Aucune analyse disponible.<br>
          Cliquez sur "Analyser" pour commencer.
        </div>
        <button class="fp-panel-btn" id="fp-analyze-btn">Analyser cette page</button>
      `;
      
      document.getElementById('fp-analyze-btn').addEventListener('click', runAnalysis);
      return;
    }
    
    if (lastResults.status === 'error') {
      contentEl.innerHTML = `
        <div class="fp-panel-error">
          ❌ Erreur: ${lastResults.message}
        </div>
        <button class="fp-panel-btn" id="fp-retry-btn">Réessayer</button>
      `;
      
      document.getElementById('fp-retry-btn').addEventListener('click', runAnalysis);
      return;
    }
    
    // Afficher les résultats
    const results = lastResults.results || [];
    const timing = lastResults.timing || {};
    
    let resultsHTML = '';
    
    if (results.length === 0) {
      resultsHTML = `
        <div class="fp-result fp-result-ok">
          ✓ Aucune affirmation douteuse détectée
        </div>
      `;
    } else {
      resultsHTML = results.map(r => {
        const verdictClass = {
          'TRUE': 'fp-verdict-true',
          'FALSE': 'fp-verdict-false',
          'PARTIALLY_TRUE': 'fp-verdict-partial',
          'NOT_VERIFIABLE': 'fp-verdict-unknown'
        }[r.verdict] || 'fp-verdict-unknown';
        
        const verdictIcon = {
          'TRUE': '✓',
          'FALSE': '✗',
          'PARTIALLY_TRUE': '~',
          'NOT_VERIFIABLE': '?'
        }[r.verdict] || '?';
        
        const sources = (r.sources || []).slice(0, 2).map(s => 
          `<a href="${s.url || '#'}" class="fp-source" target="_blank">${s.title}</a>`
        ).join('');
        
        return `
          <div class="fp-result ${verdictClass}">
            <div class="fp-result-header">
              <span class="fp-verdict-icon">${verdictIcon}</span>
              <span class="fp-verdict-label">${r.verdict}</span>
              <span class="fp-confidence">${Math.round(r.confidence * 100)}%</span>
            </div>
            <div class="fp-claim">"${r.claim_text.substring(0, 100)}..."</div>
            <div class="fp-explanation">${r.explanation || ''}</div>
            ${sources ? `<div class="fp-sources">Sources: ${sources}</div>` : ''}
          </div>
        `;
      }).join('');
    }
    
    contentEl.innerHTML = `
      <div class="fp-results">
        ${resultsHTML}
      </div>
      <div class="fp-timing">
        ⏱️ ${Math.round(timing.total_ms || 0)}ms
      </div>
      <button class="fp-panel-btn" id="fp-reanalyze-btn">Réanalyser</button>
    `;
    
    document.getElementById('fp-reanalyze-btn').addEventListener('click', runAnalysis);
  }

  // ============================================================================
  // ANALYSE PRINCIPALE
  // ============================================================================
  
  /**
   * Lance l'analyse de la page.
   */
  async function runAnalysis() {
    updateBadge('checking');
    
    try {
      const text = extractVisibleText();
      const results = await analyzeText(text);
      
      lastResults = results;
      
      if (results.status === 'error') {
        updateBadge('error', results.message);
      } else if (results.status === 'skip') {
        updateBadge('idle', results.message);
      } else {
        // Déterminer le statut basé sur les résultats
        const hasProblematic = (results.results || []).some(
          r => r.verdict === 'FALSE' || r.verdict === 'PARTIALLY_TRUE'
        );
        
        if (hasProblematic) {
          updateBadge('warning', `${results.claims_verified} claim(s)`);
        } else if (results.claims_verified > 0) {
          updateBadge('success', 'OK');
        } else {
          updateBadge('idle');
        }
      }
      
      renderPanelContent();
      
    } catch (error) {
      lastResults = { status: 'error', message: error.message };
      updateBadge('error');
      renderPanelContent();
    }
  }

  // ============================================================================
  // INITIALISATION
  // ============================================================================
  
  /**
   * Initialise l'extension.
   */
  function init() {
    // Créer le badge
    createBadge();
    updateBadge('idle');
    
    // Auto-analyse au chargement (optionnel)
    // Désactivé par défaut pour ne pas surcharger le backend
    // setTimeout(runAnalysis, 2000);
    
    console.log('[FactPulse] Extension initialisée');
  }
  
  // Attendre que le DOM soit prêt
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
  
  // Exposer pour le background script
  window.factpulse = {
    analyze: runAnalysis,
    getStatus: () => currentStatus,
    getResults: () => lastResults
  };

  // ============================================================================
  // MESSAGE LISTENER
  // ============================================================================
  
  /**
   * Écoute les messages du popup et du background script.
   */
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'analyze') {
      runAnalysis().then(() => {
        sendResponse({ 
          success: true, 
          claimsCount: lastResults?.claims_verified || 0,
          time: lastResults?.timing?.total_ms || 0
        });
      }).catch(error => {
        sendResponse({ success: false, error: error.message });
      });
      return true; // Keep channel open for async response
    }
    
    if (message.action === 'getStatus') {
      sendResponse({ 
        status: currentStatus, 
        results: lastResults 
      });
      return false;
    }
    
    return false;
  });

})();
