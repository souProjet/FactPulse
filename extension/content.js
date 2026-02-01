/**
 * FactPulse Content Script v2.0
 * 
 * Fonctionnalités avancées:
 * - Analyse uniquement du texte visible dans le viewport
 * - Détection de multiples claims dans un texte
 * - Surlignage en temps réel du texte analysé
 * - Tooltips interactifs avec détails
 * - Animations de chargement
 * 
 * Flux de données:
 * 1. Détection viewport → extraction texte visible
 * 2. Envoi au backend → animation chargement sur texte
 * 3. Réception verdicts → surlignage par claim
 * 4. Hover/click → affichage détails
 */

(function() {
  'use strict';

  // ============================================================================
  // CONFIGURATION
  // ============================================================================
  
  const CONFIG = {
    API_URL: 'http://localhost:8000',
    MAX_TEXT_LENGTH: 15000,
    MIN_TEXT_LENGTH: 30,
    DEBOUNCE_MS: 500,
    TIMEOUT_MS: 20000, // 20 secondes max
    VIEWPORT_PADDING: 100,
    SCROLL_DEBOUNCE_MS: 300,
    AUTO_ANALYZE_DELAY: 1500,
    HIGHLIGHT_CLASS_PREFIX: 'fp-hl-',
    MIN_CLAIM_LENGTH: 20,
    VIEWPORT_ONLY: false,
    DEBUG: true
  };

  // Couleurs par verdict (supporte anglais et français)
  const VERDICT_COLORS = {
    'TRUE': {
      bg: 'rgba(56, 161, 105, 0.25)',
      border: '#38a169',
      text: '#22543d',
      icon: '✓',
      label: 'Vérifié'
    },
    'VRAI': {
      bg: 'rgba(56, 161, 105, 0.25)',
      border: '#38a169',
      text: '#22543d',
      icon: '✓',
      label: 'Vérifié'
    },
    'FALSE': {
      bg: 'rgba(229, 62, 62, 0.25)',
      border: '#e53e3e',
      text: '#742a2a',
      icon: '✗',
      label: 'Faux'
    },
    'FAUX': {
      bg: 'rgba(229, 62, 62, 0.25)',
      border: '#e53e3e',
      text: '#742a2a',
      icon: '✗',
      label: 'Faux'
    },
    'PARTIALLY_TRUE': {
      bg: 'rgba(237, 137, 54, 0.25)',
      border: '#ed8936',
      text: '#744210',
      icon: '~',
      label: 'Partiellement vrai'
    },
    'PARTIELLEMENT_VRAI': {
      bg: 'rgba(237, 137, 54, 0.25)',
      border: '#ed8936',
      text: '#744210',
      icon: '~',
      label: 'Partiellement vrai'
    },
    'NOT_VERIFIABLE': {
      bg: 'rgba(113, 128, 150, 0.25)',
      border: '#718096',
      text: '#1a202c',
      icon: '?',
      label: 'Non vérifiable'
    },
    'NON_VÉRIFIABLE': {
      bg: 'rgba(113, 128, 150, 0.25)',
      border: '#718096',
      text: '#1a202c',
      icon: '?',
      label: 'Non vérifiable'
    },
    'CHECKING': {
      bg: 'rgba(102, 126, 234, 0.15)',
      border: '#667eea',
      text: '#3c366b',
      icon: '◐',
      label: 'Vérification...'
    }
  };
  
  // Mapping des verdicts vers une forme normalisée pour la logique
  function normalizeVerdict(verdict) {
    const mapping = {
      'VRAI': 'TRUE',
      'FAUX': 'FALSE',
      'PARTIELLEMENT_VRAI': 'PARTIALLY_TRUE',
      'NON_VÉRIFIABLE': 'NOT_VERIFIABLE'
    };
    return mapping[verdict] || verdict;
  }

  // ============================================================================
  // ÉTAT
  // ============================================================================
  
  let currentStatus = 'idle';
  let lastResults = null;
  let badgeElement = null;
  let panelElement = null;
  let tooltipElement = null;
  let highlightedElements = new Map(); // claim_id -> DOM elements
  let analyzedTextNodes = new Set();
  let isAnalyzing = false;
  let scrollDebounceTimer = null;
  let viewportObserver = null;
  let currentClaimId = 0;

  // ============================================================================
  // VIEWPORT DETECTION
  // ============================================================================

  /**
   * Vérifie si un élément est visible dans le viewport.
   */
  function isInViewport(element, padding = CONFIG.VIEWPORT_PADDING) {
    const rect = element.getBoundingClientRect();
    return (
      rect.top < window.innerHeight + padding &&
      rect.bottom > -padding &&
      rect.left < window.innerWidth + padding &&
      rect.right > -padding
    );
  }

  /**
   * Vérifie si un rectangle est dans le viewport.
   */
  function rectInViewport(rect, padding = CONFIG.VIEWPORT_PADDING) {
    return (
      rect.top < window.innerHeight + padding &&
      rect.bottom > -padding &&
      rect.left < window.innerWidth + padding &&
      rect.right > -padding
    );
  }

  // ============================================================================
  // EXTRACTION DE TEXTE VISIBLE
  // ============================================================================
  
  /**
   * Structure pour stocker les infos d'un noeud texte.
   */
  class TextNodeInfo {
    constructor(node, text, rect) {
      this.node = node;
      this.text = text;
      this.rect = rect;
      this.element = node.parentElement;
    }
  }

  /**
   * Log de debug conditionnel
   */
  function debug(...args) {
    if (CONFIG.DEBUG) {
      console.log('[FactPulse]', ...args);
    }
  }

  /**
   * Extrait le texte de la page de manière intelligente.
   * Cible les balises de contenu sémantique et filtre le bruit.
   */
  function extractPageText() {
    // Balises à ignorer complètement
    const ignoreTags = new Set([
      'SCRIPT', 'STYLE', 'NOSCRIPT', 'IFRAME', 'SVG', 'PATH', 'CANVAS',
      'VIDEO', 'AUDIO', 'IMG', 'BR', 'HR', 'META', 'LINK', 'INPUT',
      'BUTTON', 'SELECT', 'TEXTAREA', 'FORM', 'LABEL', 'NAV', 'HEADER',
      'FOOTER', 'ASIDE', 'MENU', 'MENUITEM', 'FIGURE', 'FIGCAPTION'
    ]);
    
    // Balises de contenu sémantique à prioriser
    const contentTags = new Set([
      'P', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'BLOCKQUOTE', 
      'LI', 'TD', 'TH', 'CAPTION', 'DT', 'DD', 'PRE', 'CITE'
    ]);
    
    // Balises de liens/boutons à ignorer pour le texte seul
    const interactiveTags = new Set(['A', 'BUTTON', 'INPUT', 'SELECT']);
    
    // Classes/IDs à ignorer (navigation, publicités, widgets)
    const ignorePatterns = [
      'nav', 'menu', 'sidebar', 'footer', 'header', 'breadcrumb',
      'advertisement', 'ad-', 'ads-', 'social', 'share', 'comment',
      'related', 'widget', 'popup', 'modal', 'cookie', 'banner',
      'factpulse', 'newsletter', 'subscribe', 'signup', 'login'
    ];
    
    const ignoreIds = new Set(['factpulse-badge', 'factpulse-panel', 'factpulse-tooltip']);
    
    function matchesIgnorePattern(el) {
      const className = (el.className && typeof el.className === 'string') ? el.className.toLowerCase() : '';
      const id = el.id ? el.id.toLowerCase() : '';
      const role = el.getAttribute('role') || '';
      
      // Ignorer les rôles de navigation
      if (['navigation', 'banner', 'contentinfo', 'complementary', 'search', 'menu', 'menubar'].includes(role)) {
        return true;
      }
      
      for (const pattern of ignorePatterns) {
        if (className.includes(pattern) || id.includes(pattern)) {
          return true;
        }
      }
      return false;
    }
    
    function shouldIgnoreElement(el) {
      if (!el || !el.tagName) return true;
      if (ignoreTags.has(el.tagName)) return true;
      if (el.id && ignoreIds.has(el.id)) return true;
      if (matchesIgnorePattern(el)) return true;
      
      // Vérifier si caché
      try {
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return true;
      } catch (e) {
        // Ignorer les erreurs de style
      }
      
      return false;
    }
    
    /**
     * Vérifie si le texte ressemble à du contenu utile (pas un lien/bouton seul)
     */
    function isUsefulText(text, parentTag) {
      const trimmed = text.trim();
      
      // Trop court
      if (trimmed.length < 15) return false;
      
      // Doit contenir au moins 3 mots
      const words = trimmed.split(/\s+/).filter(w => w.length > 2);
      if (words.length < 3) return false;
      
      // Éviter les textes qui ne sont que des URLs
      if (/^https?:\/\//.test(trimmed)) return false;
      
      // Éviter les textes avec trop de caractères spéciaux (probablement du code/data)
      const specialChars = (trimmed.match(/[{}\[\]<>|\\`~@#$%^&*]/g) || []).length;
      if (specialChars > trimmed.length * 0.1) return false;
      
      // Pour les liens <a>, ignorer sauf s'ils font partie d'un paragraphe
      if (parentTag === 'A' && words.length < 6) return false;
      
      return true;
    }
    
    /**
     * Extrait le texte des balises de contenu sémantique
     */
    function extractFromContentElement(el) {
      // Récupérer tout le texte de l'élément
      let text = '';
      
      // Parcourir les enfants et construire le texte
      for (const child of el.childNodes) {
        if (child.nodeType === Node.TEXT_NODE) {
          text += child.textContent;
        } else if (child.nodeType === Node.ELEMENT_NODE) {
          // Inclure le texte des enfants sauf certains éléments interactifs isolés
          if (!ignoreTags.has(child.tagName)) {
            // Pour les liens, on inclut le texte mais pas s'ils sont le seul contenu
            text += child.textContent;
          }
        }
      }
      
      return text.replace(/\s+/g, ' ').trim();
    }
    
    function extractFromElement(el, texts, depth = 0) {
      if (shouldIgnoreElement(el)) return;
      if (depth > 20) return; // Protection contre récursion trop profonde
      
      // Si c'est une balise de contenu sémantique, extraire le texte directement
      if (contentTags.has(el.tagName)) {
        const text = extractFromContentElement(el);
        if (isUsefulText(text, el.tagName)) {
          texts.push(text);
        }
        return; // Ne pas récurser dans les enfants (déjà traités)
      }
      
      // Sinon, récurser dans les enfants
      for (const child of el.childNodes) {
        if (child.nodeType === Node.ELEMENT_NODE) {
          extractFromElement(child, texts, depth + 1);
        }
      }
    }
    
    // Chercher d'abord le contenu principal
    const mainSelectors = [
      'article', 
      'main', 
      '[role="main"]', 
      '.content:not(.nav-content):not(.header-content)', 
      '.post-content',
      '.post-body',
      '.article-body', 
      '.article-content',
      '.entry-content', 
      '.story-body',
      '.text-content',
      '#content:not(#nav-content)', 
      '#main',
      '#article',
      '.prose'
    ];
    let targetElement = null;
    
    for (const selector of mainSelectors) {
      try {
        targetElement = document.querySelector(selector);
        if (targetElement && targetElement.textContent.trim().length > 100) {
          debug('Contenu principal trouvé:', selector);
          break;
        }
        targetElement = null;
      } catch (e) {
        // Sélecteur invalide, ignorer
      }
    }
    
    // Fallback vers body
    if (!targetElement) {
      targetElement = document.body;
      debug('Utilisation de body comme cible');
    }
    
    const texts = [];
    extractFromElement(targetElement, texts, 0);
    
    // Si on n'a pas trouvé assez de contenu, fallback plus permissif
    if (texts.length === 0 || texts.join(' ').length < 100) {
      debug('Contenu insuffisant, fallback vers extraction permissive');
      texts.length = 0;
      
      // Chercher tous les <p> et headings de la page
      const allContentElements = document.querySelectorAll('p, h1, h2, h3, h4, h5, h6, blockquote, li');
      for (const el of allContentElements) {
        if (!shouldIgnoreElement(el) && !shouldIgnoreElement(el.parentElement)) {
          const text = el.textContent.replace(/\s+/g, ' ').trim();
          if (isUsefulText(text, el.tagName)) {
            texts.push(text);
          }
        }
      }
    }
    
    const fullText = texts.join('\n\n').replace(/\n{3,}/g, '\n\n').trim();
    debug('Texte extrait:', fullText.length, 'caractères,', texts.length, 'paragraphes');
    debug('Aperçu:', fullText.substring(0, 300) + '...');
    
    return fullText;
  }

  /**
   * Extrait les noeuds de texte pour le surlignage.
   * Filtre les éléments non pertinents comme la navigation.
   */
  function extractTextNodesForHighlighting() {
    const ignoreTags = new Set([
      'SCRIPT', 'STYLE', 'NOSCRIPT', 'IFRAME', 'SVG', 'PATH', 'CANVAS',
      'INPUT', 'BUTTON', 'SELECT', 'TEXTAREA', 'NAV', 'HEADER', 'FOOTER',
      'ASIDE', 'MENU', 'MENUITEM'
    ]);
    
    // Balises de contenu à cibler pour le highlighting
    const contentParentTags = new Set([
      'P', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'BLOCKQUOTE', 
      'LI', 'TD', 'TH', 'CAPTION', 'DT', 'DD', 'PRE', 'CITE',
      'SPAN', 'EM', 'STRONG', 'B', 'I', 'MARK', 'DIV', 'SECTION', 'ARTICLE'
    ]);
    
    const ignorePatterns = [
      'nav', 'menu', 'sidebar', 'footer', 'header', 'breadcrumb',
      'advertisement', 'ad-', 'social', 'share', 'comment',
      'related', 'widget', 'popup', 'modal', 'cookie', 'banner',
      'factpulse', 'newsletter', 'subscribe', 'signup', 'login'
    ];
    
    function hasIgnorePattern(el) {
      if (!el) return false;
      const className = (el.className && typeof el.className === 'string') ? el.className.toLowerCase() : '';
      const id = el.id ? el.id.toLowerCase() : '';
      const role = el.getAttribute ? (el.getAttribute('role') || '') : '';
      
      if (['navigation', 'banner', 'contentinfo', 'complementary', 'search', 'menu'].includes(role)) {
        return true;
      }
      
      for (const pattern of ignorePatterns) {
        if (className.includes(pattern) || id.includes(pattern)) {
          return true;
        }
      }
      return false;
    }
    
    function isInContentArea(el) {
      // Vérifier les ancêtres jusqu'à 5 niveaux
      let current = el;
      for (let i = 0; i < 5 && current; i++) {
        if (hasIgnorePattern(current)) return false;
        if (ignoreTags.has(current.tagName)) return false;
        current = current.parentElement;
      }
      return true;
    }
    
    const textNodes = [];
    
    const walker = document.createTreeWalker(
      document.body,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode: (node) => {
          const parent = node.parentElement;
          if (!parent) return NodeFilter.FILTER_REJECT;
          if (ignoreTags.has(parent.tagName)) return NodeFilter.FILTER_REJECT;
          
          // Ignorer nos éléments
          if (parent.id && parent.id.startsWith('factpulse')) return NodeFilter.FILTER_REJECT;
          if (parent.className && typeof parent.className === 'string' && 
              parent.className.includes('factpulse')) return NodeFilter.FILTER_REJECT;
          
          const text = node.textContent.trim();
          if (text.length < 10) return NodeFilter.FILTER_REJECT;
          
          // Vérifier si dans une zone de contenu
          if (!isInContentArea(parent)) return NodeFilter.FILTER_REJECT;
          
          // Ignorer les liens isolés (< 30 caractères et parent direct est <a>)
          if (parent.tagName === 'A' && text.length < 30) {
            return NodeFilter.FILTER_REJECT;
          }
          
          return NodeFilter.FILTER_ACCEPT;
        }
      }
    );
    
    while (walker.nextNode()) {
      const node = walker.currentNode;
      const text = node.textContent.trim();
      
      // Si viewport only, vérifier la position
      if (CONFIG.VIEWPORT_ONLY) {
        try {
          const range = document.createRange();
          range.selectNodeContents(node);
          const rect = range.getClientRects()[0];
          if (rect && rectInViewport(rect)) {
            textNodes.push(new TextNodeInfo(node, text, rect));
          }
        } catch (e) {
          // Ignorer les erreurs de range
        }
      } else {
        textNodes.push(new TextNodeInfo(node, text, null));
      }
    }
    
    debug('Noeuds texte trouvés pour highlighting:', textNodes.length);
    return textNodes;
  }

  /**
   * Ancienne fonction conservée pour compatibilité
   */
  function extractVisibleTextNodes() {
    return extractTextNodesForHighlighting();
  }

  /**
   * Extrait le texte visible et retourne les infos pour le highlighting.
   */
  function extractVisibleContent() {
    const text = extractPageText();
    const textNodes = extractTextNodesForHighlighting();
    
    return {
      text: text.substring(0, CONFIG.MAX_TEXT_LENGTH),
      textNodes,
      paragraphs: [],
      nodeMap: new Map()
    };
  }

  // ============================================================================
  // TEXT HIGHLIGHTING
  // ============================================================================

  /**
   * Crée un élément de surlignage pour un claim.
   */
  function createHighlightWrapper(claimId, verdict, claimText) {
    const colors = VERDICT_COLORS[verdict] || VERDICT_COLORS['NOT_VERIFIABLE'];
    
    const wrapper = document.createElement('span');
    wrapper.className = `factpulse-highlight fp-hl-${verdict.toLowerCase()}`;
    wrapper.dataset.claimId = claimId;
    wrapper.dataset.verdict = verdict;
    wrapper.dataset.claimText = claimText;
    
    wrapper.style.cssText = `
      background-color: ${colors.bg};
      border-bottom: 2px solid ${colors.border};
      border-radius: 2px;
      padding: 1px 2px;
      margin: 0 -2px;
      cursor: pointer;
      transition: all 0.2s ease;
      position: relative;
    `;
    
    return wrapper;
  }

  /**
   * Recherche une sous-chaîne dans un texte avec tolérance.
   */
  function fuzzyFind(text, search, tolerance = 0.8) {
    const normalizedText = text.toLowerCase().replace(/\s+/g, ' ').trim();
    const normalizedSearch = search.toLowerCase().replace(/\s+/g, ' ').trim();
    
    // Recherche exacte d'abord
    const exactIndex = normalizedText.indexOf(normalizedSearch);
    if (exactIndex !== -1) {
      return { start: exactIndex, end: exactIndex + search.length, exact: true };
    }
    
    // Recherche avec différentes longueurs de début
    for (let len = Math.min(40, normalizedSearch.length); len >= 15; len -= 5) {
      const searchStart = normalizedSearch.substring(0, len);
      const partialIndex = normalizedText.indexOf(searchStart);
      if (partialIndex !== -1) {
        return { start: partialIndex, end: partialIndex + Math.min(search.length, 150), exact: false };
      }
    }
    
    // Recherche par mots-clés
    const words = normalizedSearch.split(' ').filter(w => w.length > 4);
    if (words.length >= 2) {
      const firstWord = words[0];
      const idx = normalizedText.indexOf(firstWord);
      if (idx !== -1) {
        return { start: idx, end: idx + Math.min(80, normalizedSearch.length), exact: false };
      }
    }
    
    return null;
  }

  /**
   * Trouve et surligne un claim dans le DOM.
   */
  function highlightClaimInDOM(claimText, verdict, claimId, result) {
    const textNodes = extractTextNodesForHighlighting();
    
    debug('Recherche du claim:', claimText.substring(0, 50) + '...');
    debug('Nombre de noeuds à parcourir:', textNodes.length);
    
    // Normaliser le claim pour la recherche
    const normalizedClaim = claimText.toLowerCase().replace(/\s+/g, ' ').trim();
    
    // Essayer de trouver une correspondance
    for (const nodeInfo of textNodes) {
      const nodeText = nodeInfo.text;
      const normalizedNodeText = nodeText.toLowerCase().replace(/\s+/g, ' ').trim();
      
      // Vérifier si le claim est dans ce noeud
      const matchResult = fuzzyFind(normalizedNodeText, normalizedClaim);
      
      if (matchResult) {
        debug('Match trouvé dans noeud:', normalizedNodeText.substring(0, 50) + '...');
        highlightTextNode(nodeInfo.node, claimText, verdict, claimId, result);
        return true;
      }
    }
    
    // Pas de correspondance exacte, chercher dans tout le document
    debug('Pas de match direct, recherche étendue...');
    
    // Utiliser window.find comme fallback (plus lent mais plus fiable)
    if (window.find && claimText.length > 20) {
      const searchTerm = claimText.substring(0, 40);
      // Cette approche ne modifie pas le DOM, juste pour le debug
      debug('Terme recherché:', searchTerm);
    }
    
    return false;
  }

  /**
   * Surligne un texte spécifique dans un noeud.
   */
  function highlightTextNode(textNode, claimText, verdict, claimId, result) {
    const parent = textNode.parentElement;
    if (!parent || parent.classList.contains('factpulse-highlight')) return;
    
    const nodeText = textNode.textContent;
    const normalizedNodeText = nodeText.toLowerCase();
    const normalizedClaim = claimText.toLowerCase().substring(0, 50);
    
    const startIndex = normalizedNodeText.indexOf(normalizedClaim);
    
    if (startIndex === -1) {
      // Fallback: surligner tout le noeud si on ne trouve pas exactement
      wrapEntireNode(textNode, verdict, claimId, result);
      return;
    }
    
    // Créer les fragments
    const before = nodeText.substring(0, startIndex);
    const match = nodeText.substring(startIndex, startIndex + claimText.length);
    const after = nodeText.substring(startIndex + claimText.length);
    
    const fragment = document.createDocumentFragment();
    
    if (before) {
      fragment.appendChild(document.createTextNode(before));
    }
    
    const wrapper = createHighlightWrapper(claimId, verdict, claimText);
    wrapper.textContent = match || nodeText.substring(startIndex, startIndex + 100);
    wrapper.addEventListener('mouseenter', (e) => showTooltip(e, result));
    wrapper.addEventListener('mouseleave', hideTooltip);
    wrapper.addEventListener('click', (e) => showDetailPanel(e, result));
    
    fragment.appendChild(wrapper);
    
    if (after) {
      fragment.appendChild(document.createTextNode(after));
    }
    
    // Remplacer le noeud texte
    parent.replaceChild(fragment, textNode);
    
    // Stocker la référence
    if (!highlightedElements.has(claimId)) {
      highlightedElements.set(claimId, []);
    }
    highlightedElements.get(claimId).push(wrapper);
  }

  /**
   * Enveloppe un noeud entier avec un surlignage.
   */
  function wrapEntireNode(textNode, verdict, claimId, result) {
    const parent = textNode.parentElement;
    if (!parent || parent.classList.contains('factpulse-highlight')) return;
    
    const wrapper = createHighlightWrapper(claimId, verdict, result.claim_text);
    wrapper.textContent = textNode.textContent;
    wrapper.addEventListener('mouseenter', (e) => showTooltip(e, result));
    wrapper.addEventListener('mouseleave', hideTooltip);
    wrapper.addEventListener('click', (e) => showDetailPanel(e, result));
    
    parent.replaceChild(wrapper, textNode);
    
    if (!highlightedElements.has(claimId)) {
      highlightedElements.set(claimId, []);
    }
    highlightedElements.get(claimId).push(wrapper);
  }

  /**
   * Affiche l'animation de chargement sur les textes visibles.
   */
  function showLoadingHighlights() {
    const textNodes = extractVisibleTextNodes();
    
    // Surligner quelques éléments aléatoires pour montrer l'activité
    const toHighlight = textNodes.slice(0, Math.min(5, textNodes.length));
    
    for (const nodeInfo of toHighlight) {
      const parent = nodeInfo.node.parentElement;
      if (!parent || parent.classList.contains('factpulse-highlight')) continue;
      
      const wrapper = document.createElement('span');
      wrapper.className = 'factpulse-loading-highlight';
      wrapper.textContent = nodeInfo.node.textContent;
      
      parent.replaceChild(wrapper, nodeInfo.node);
      analyzedTextNodes.add(wrapper);
    }
  }

  /**
   * Supprime les highlights de chargement.
   */
  function removeLoadingHighlights() {
    const loadingElements = document.querySelectorAll('.factpulse-loading-highlight');
    
    for (const el of loadingElements) {
      const textNode = document.createTextNode(el.textContent);
      el.parentNode.replaceChild(textNode, el);
    }
    
    analyzedTextNodes.clear();
  }

  /**
   * Supprime tous les highlights.
   */
  function clearAllHighlights() {
    removeLoadingHighlights();
    
    const highlights = document.querySelectorAll('.factpulse-highlight');
    for (const el of highlights) {
      const textNode = document.createTextNode(el.textContent);
      el.parentNode.replaceChild(textNode, el);
    }
    
    highlightedElements.clear();
  }

  // ============================================================================
  // TOOLTIP
  // ============================================================================

  /**
   * Crée le tooltip si nécessaire.
   */
  function createTooltip() {
    if (tooltipElement) return tooltipElement;
    
    tooltipElement = document.createElement('div');
    tooltipElement.id = 'factpulse-tooltip';
    tooltipElement.className = 'factpulse-tooltip';
    document.body.appendChild(tooltipElement);
    
    return tooltipElement;
  }

  /**
   * Affiche le tooltip pour un résultat.
   */
  function showTooltip(event, result) {
    const tooltip = createTooltip();
    const colors = VERDICT_COLORS[result.verdict] || VERDICT_COLORS['NOT_VERIFIABLE'];
    
    const confidencePercent = Math.round(result.confidence * 100);
    
    tooltip.innerHTML = `
      <div class="fp-tooltip-header" style="border-left: 3px solid ${colors.border};">
        <span class="fp-tooltip-icon">${colors.icon}</span>
        <span class="fp-tooltip-verdict">${colors.label}</span>
        <span class="fp-tooltip-confidence">${confidencePercent}%</span>
      </div>
      <div class="fp-tooltip-body">
        <p class="fp-tooltip-explanation">${result.explanation || 'Analyse en cours...'}</p>
        ${result.sources && result.sources.length > 0 ? `
          <div class="fp-tooltip-sources">
            <span class="fp-tooltip-sources-label">Sources:</span>
            ${result.sources.slice(0, 2).map(s => 
              `<a href="${s.url || '#'}" class="fp-tooltip-source" target="_blank">${s.title || s.url}</a>`
            ).join('')}
          </div>
        ` : ''}
      </div>
      <div class="fp-tooltip-footer">
        Cliquez pour plus de détails
      </div>
    `;
    
    // Positionner le tooltip
    const rect = event.target.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();
    
    let left = rect.left + (rect.width / 2) - 150;
    let top = rect.bottom + 10;
    
    // Ajuster si dépasse à droite
    if (left + 300 > window.innerWidth) {
      left = window.innerWidth - 320;
    }
    if (left < 10) left = 10;
    
    // Ajuster si dépasse en bas
    if (top + 200 > window.innerHeight) {
      top = rect.top - 210;
    }
    
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
    tooltip.classList.add('fp-tooltip-visible');
  }

  /**
   * Cache le tooltip.
   */
  function hideTooltip() {
    if (tooltipElement) {
      tooltipElement.classList.remove('fp-tooltip-visible');
    }
  }

  // ============================================================================
  // DETAIL PANEL
  // ============================================================================

  /**
   * Affiche le panneau de détails pour un claim.
   */
  function showDetailPanel(event, result) {
    event.stopPropagation();
    hideTooltip();
    
    const panel = createPanel();
    const contentEl = panel.querySelector('.fp-panel-content');
    const colors = VERDICT_COLORS[result.verdict] || VERDICT_COLORS['NOT_VERIFIABLE'];
    
    contentEl.innerHTML = `
      <div class="fp-detail-card">
        <div class="fp-detail-verdict" style="background: ${colors.bg}; border-color: ${colors.border};">
          <span class="fp-detail-icon" style="background: ${colors.border};">${colors.icon}</span>
          <div class="fp-detail-verdict-info">
            <span class="fp-detail-label">${colors.label}</span>
            <span class="fp-detail-confidence">Confiance: ${Math.round(result.confidence * 100)}%</span>
          </div>
        </div>
        
        <div class="fp-detail-section">
          <h4>Affirmation analysée</h4>
          <p class="fp-detail-claim">"${result.claim_text}"</p>
        </div>
        
        <div class="fp-detail-section">
          <h4>Explication</h4>
          <p class="fp-detail-explanation">${result.explanation || 'Aucune explication disponible.'}</p>
        </div>
        
        ${result.sources && result.sources.length > 0 ? `
          <div class="fp-detail-section">
            <h4>Sources</h4>
            <div class="fp-detail-sources">
              ${result.sources.map(s => `
                <a href="${s.url || '#'}" class="fp-detail-source" target="_blank">
                  <span class="fp-source-icon">🔗</span>
                  <span class="fp-source-title">${s.title || s.url}</span>
                </a>
              `).join('')}
            </div>
          </div>
        ` : ''}
        
        <div class="fp-detail-actions">
          <button class="fp-action-btn fp-action-feedback" data-claim="${encodeURIComponent(result.claim_text)}">
            ⚑ Signaler une erreur
          </button>
        </div>
      </div>
    `;
    
    // Handler pour le feedback
    const feedbackBtn = contentEl.querySelector('.fp-action-feedback');
    if (feedbackBtn) {
      feedbackBtn.addEventListener('click', () => showFeedbackForm(result));
    }
    
    panel.classList.add('fp-panel-visible');
  }

  /**
   * Affiche le formulaire de feedback.
   */
  function showFeedbackForm(result) {
    const panel = panelElement;
    const contentEl = panel.querySelector('.fp-panel-content');
    
    contentEl.innerHTML = `
      <div class="fp-feedback-form">
        <h4>Signaler une erreur</h4>
        <p class="fp-feedback-claim">"${result.claim_text.substring(0, 100)}..."</p>
        
        <div class="fp-feedback-field">
          <label>Le verdict devrait être :</label>
          <select id="fp-feedback-verdict">
            <option value="TRUE">Vrai</option>
            <option value="FALSE">Faux</option>
            <option value="PARTIALLY_TRUE">Partiellement vrai</option>
            <option value="NOT_VERIFIABLE">Non vérifiable</option>
          </select>
        </div>
        
        <div class="fp-feedback-field">
          <label>Commentaire (optionnel) :</label>
          <textarea id="fp-feedback-comment" rows="3" placeholder="Expliquez pourquoi..."></textarea>
        </div>
        
        <div class="fp-feedback-actions">
          <button class="fp-action-btn" id="fp-feedback-cancel">Annuler</button>
          <button class="fp-action-btn fp-action-primary" id="fp-feedback-submit">Envoyer</button>
        </div>
      </div>
    `;
    
    document.getElementById('fp-feedback-cancel').addEventListener('click', () => {
      renderPanelContent();
    });
    
    document.getElementById('fp-feedback-submit').addEventListener('click', async () => {
      const actualVerdict = document.getElementById('fp-feedback-verdict').value;
      const comment = document.getElementById('fp-feedback-comment').value;
      
      await submitFeedback(result, actualVerdict, comment);
    });
  }

  /**
   * Envoie le feedback au backend.
   */
  async function submitFeedback(result, actualVerdict, comment) {
    try {
      const response = await fetch(`${CONFIG.API_URL}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          claim_text: result.claim_text,
          predicted_verdict: result.verdict,
          actual_verdict: actualVerdict,
          confidence: result.confidence,
          feedback: comment
        })
      });
      
      if (response.ok) {
        showNotification('Merci pour votre feedback !', 'success');
      } else {
        showNotification('Erreur lors de l\'envoi', 'error');
      }
    } catch (error) {
      showNotification('Erreur de connexion', 'error');
    }
    
    hidePanel();
  }

  // ============================================================================
  // NOTIFICATION
  // ============================================================================

  function showNotification(message, type = 'info') {
    const notif = document.createElement('div');
    notif.className = `factpulse-notification fp-notif-${type}`;
    notif.textContent = message;
    document.body.appendChild(notif);
    
    setTimeout(() => notif.classList.add('fp-notif-visible'), 10);
    setTimeout(() => {
      notif.classList.remove('fp-notif-visible');
      setTimeout(() => notif.remove(), 300);
    }, 3000);
  }

  // ============================================================================
  // COMMUNICATION API
  // ============================================================================
  
  /**
   * Envoie le texte au backend pour analyse.
   */
  async function analyzeText(text) {
    if (text.length < CONFIG.MIN_TEXT_LENGTH) {
      debug('Texte trop court pour analyse:', text.length);
      return { status: 'skip', message: 'Texte trop court' };
    }
    
    debug('Envoi au backend:', CONFIG.API_URL + '/analyze');
    debug('Longueur du texte:', text.length);
    
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
      
      debug('Réponse HTTP:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        debug('Erreur response:', errorText);
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }
      
      const data = await response.json();
      debug('Données reçues:', data);
      return data;
    } catch (error) {
      clearTimeout(timeout);
      
      debug('Erreur fetch:', error.name, error.message);
      
      if (error.name === 'AbortError') {
        return { status: 'error', message: 'Timeout - le serveur met trop de temps' };
      }
      
      if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        return { status: 'error', message: 'Serveur non disponible. Démarrez le backend.' };
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
      <div class="fp-badge-count"></div>
    `;
    
    badgeElement.addEventListener('click', togglePanel);
    
    document.body.appendChild(badgeElement);
    return badgeElement;
  }
  
  /**
   * Met à jour le statut du badge.
   */
  function updateBadge(status, message = '', count = 0) {
    currentStatus = status;
    
    const badge = createBadge();
    badge.className = `fp-badge fp-badge-${status}`;
    
    const iconEl = badge.querySelector('.fp-badge-icon');
    const textEl = badge.querySelector('.fp-badge-text');
    const countEl = badge.querySelector('.fp-badge-count');
    
    const icons = {
      idle: '○',
      checking: '◐',
      success: '✓',
      warning: '⚠',
      error: '✗'
    };
    
    const texts = {
      idle: 'FactPulse',
      checking: 'Analyse...',
      success: 'Vérifié',
      warning: 'Attention',
      error: 'Erreur'
    };
    
    iconEl.textContent = icons[status] || '○';
    textEl.textContent = message || texts[status] || 'FactPulse';
    
    // Afficher le compteur
    if (count > 0) {
      countEl.textContent = count;
      countEl.style.display = 'flex';
    } else {
      countEl.style.display = 'none';
    }
    
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
          <div class="fp-empty-icon">🔍</div>
          <p>Prêt à analyser</p>
          <p class="fp-empty-hint">Cliquez ci-dessous pour vérifier le texte visible à l'écran</p>
        </div>
        <button class="fp-panel-btn" id="fp-analyze-btn">
          <span class="fp-btn-icon">⚡</span>
          Analyser cette page
        </button>
      `;
      
      document.getElementById('fp-analyze-btn').addEventListener('click', runAnalysis);
      return;
    }
    
    if (lastResults.status === 'error') {
      contentEl.innerHTML = `
        <div class="fp-panel-error">
          <div class="fp-error-icon">❌</div>
          <p>Erreur: ${lastResults.message}</p>
        </div>
        <button class="fp-panel-btn" id="fp-retry-btn">Réessayer</button>
      `;
      
      document.getElementById('fp-retry-btn').addEventListener('click', runAnalysis);
      return;
    }
    
    // Afficher le résumé des résultats
    const results = lastResults.results || [];
    const timing = lastResults.timing || {};
    
    const verdictCounts = {
      TRUE: 0,
      FALSE: 0,
      PARTIALLY_TRUE: 0,
      NOT_VERIFIABLE: 0
    };
    
    for (const r of results) {
      const normalized = normalizeVerdict(r.verdict);
      if (verdictCounts.hasOwnProperty(normalized)) {
        verdictCounts[normalized]++;
      }
    }
    
    let resultsHTML = '';
    
    if (results.length === 0) {
      resultsHTML = `
        <div class="fp-result fp-result-ok">
          <div class="fp-result-icon">✓</div>
          <p>Aucune affirmation vérifiable détectée</p>
        </div>
      `;
    } else {
      // Résumé
      resultsHTML = `
        <div class="fp-summary">
          <div class="fp-summary-item fp-summary-true">
            <span class="fp-summary-count">${verdictCounts.TRUE}</span>
            <span class="fp-summary-label">Vrai</span>
          </div>
          <div class="fp-summary-item fp-summary-false">
            <span class="fp-summary-count">${verdictCounts.FALSE}</span>
            <span class="fp-summary-label">Faux</span>
          </div>
          <div class="fp-summary-item fp-summary-partial">
            <span class="fp-summary-count">${verdictCounts.PARTIALLY_TRUE}</span>
            <span class="fp-summary-label">Partiel</span>
          </div>
          <div class="fp-summary-item fp-summary-unknown">
            <span class="fp-summary-count">${verdictCounts.NOT_VERIFIABLE}</span>
            <span class="fp-summary-label">Inconnu</span>
          </div>
        </div>
        
        <div class="fp-results-list">
          <h4>Affirmations détectées (${results.length})</h4>
          ${results.map((r, i) => {
            const colors = VERDICT_COLORS[r.verdict] || VERDICT_COLORS['NOT_VERIFIABLE'];
            return `
              <div class="fp-result-item" data-index="${i}">
                <div class="fp-result-icon-small" style="background: ${colors.border};">${colors.icon}</div>
                <div class="fp-result-content">
                  <div class="fp-result-claim-short">${r.claim_text.substring(0, 80)}${r.claim_text.length > 80 ? '...' : ''}</div>
                  <div class="fp-result-meta">
                    <span class="fp-result-verdict" style="color: ${colors.border};">${colors.label}</span>
                    <span class="fp-result-conf">${Math.round(r.confidence * 100)}%</span>
                  </div>
                </div>
              </div>
            `;
          }).join('')}
        </div>
      `;
    }
    
    contentEl.innerHTML = `
      <div class="fp-results">
        ${resultsHTML}
      </div>
      <div class="fp-panel-footer">
        <div class="fp-timing">⏱️ ${Math.round(timing.total_ms || 0)}ms</div>
        <button class="fp-panel-btn-small" id="fp-clear-btn">Effacer</button>
        <button class="fp-panel-btn-small fp-btn-primary" id="fp-reanalyze-btn">Réanalyser</button>
      </div>
    `;
    
    // Event listeners
    document.getElementById('fp-reanalyze-btn').addEventListener('click', runAnalysis);
    document.getElementById('fp-clear-btn').addEventListener('click', () => {
      clearAllHighlights();
      lastResults = null;
      updateBadge('idle');
      renderPanelContent();
    });
    
    // Click sur un résultat pour scroll vers le highlight
    const resultItems = contentEl.querySelectorAll('.fp-result-item');
    resultItems.forEach((item, i) => {
      item.addEventListener('click', () => {
        const result = results[i];
        scrollToHighlight(result.claim_text);
      });
    });
  }

  /**
   * Scroll vers un highlight spécifique.
   */
  function scrollToHighlight(claimText) {
    const highlights = document.querySelectorAll('.factpulse-highlight');
    
    for (const hl of highlights) {
      if (hl.dataset.claimText === claimText || 
          hl.textContent.toLowerCase().includes(claimText.substring(0, 30).toLowerCase())) {
        hl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        
        // Flash effect
        hl.classList.add('fp-hl-flash');
        setTimeout(() => hl.classList.remove('fp-hl-flash'), 1000);
        
        break;
      }
    }
    
    hidePanel();
  }

  // ============================================================================
  // ANALYSE PRINCIPALE
  // ============================================================================
  
  /**
   * Lance l'analyse de la page.
   */
  async function runAnalysis() {
    if (isAnalyzing) return;
    
    isAnalyzing = true;
    debug('=== Début de l\'analyse ===');
    updateBadge('checking');
    clearAllHighlights();
    showLoadingHighlights();
    
    try {
      // Extraire le texte visible
      const { text, textNodes } = extractVisibleContent();
      
      debug('Texte extrait:', text.length, 'caractères');
      debug('Noeuds texte:', textNodes.length);
      
      if (text.length < CONFIG.MIN_TEXT_LENGTH) {
        debug('Texte trop court:', text.length, '<', CONFIG.MIN_TEXT_LENGTH);
        removeLoadingHighlights();
        lastResults = { status: 'skip', message: 'Pas assez de texte' };
        updateBadge('idle', 'Pas de texte');
        renderPanelContent();
        isAnalyzing = false;
        return;
      }
      
      debug('Envoi au backend...');
      debug('Texte envoyé (aperçu):', text.substring(0, 300) + '...');
      
      // Analyser le texte
      const results = await analyzeText(text);
      
      debug('Résultats reçus:', results);
      
      removeLoadingHighlights();
      lastResults = results;
      
      if (results.status === 'error') {
        debug('Erreur:', results.message);
        updateBadge('error', results.message);
      } else if (results.status === 'skip') {
        debug('Skip:', results.message);
        updateBadge('idle', results.message);
      } else {
        // Surligner chaque claim détecté
        const claims = results.results || [];
        debug('Claims détectés:', claims.length);
        
        let problematicCount = 0;
        let highlightedCount = 0;
        
        for (let i = 0; i < claims.length; i++) {
          const claim = claims[i];
          const claimId = `claim-${++currentClaimId}`;
          const normalizedVerdict = normalizeVerdict(claim.verdict);
          
          debug(`Claim ${i + 1}:`, claim.claim_text.substring(0, 60) + '...');
          debug(`  Verdict: ${claim.verdict} (normalisé: ${normalizedVerdict}), Confiance: ${claim.confidence}`);
          
          const highlighted = highlightClaimInDOM(claim.claim_text, claim.verdict, claimId, claim);
          if (highlighted) highlightedCount++;
          
          // Utiliser le verdict normalisé pour la détection des problèmes
          if (normalizedVerdict === 'FALSE' || normalizedVerdict === 'PARTIALLY_TRUE') {
            problematicCount++;
          }
        }
        
        debug(`Surlignages effectués: ${highlightedCount}/${claims.length}`);
        
        // Mettre à jour le badge
        if (problematicCount > 0) {
          updateBadge('warning', `${problematicCount} problème(s)`, claims.length);
        } else if (claims.length > 0) {
          updateBadge('success', 'Vérifié', claims.length);
        } else {
          updateBadge('idle', 'RAS');
        }
      }
      
      renderPanelContent();
      
    } catch (error) {
      removeLoadingHighlights();
      lastResults = { status: 'error', message: error.message };
      updateBadge('error');
      renderPanelContent();
    }
    
    isAnalyzing = false;
  }

  // ============================================================================
  // SCROLL OBSERVER
  // ============================================================================

  /**
   * Observe le scroll pour réanalyser si besoin.
   */
  function setupScrollObserver() {
    let lastScrollTop = window.scrollY;
    
    window.addEventListener('scroll', () => {
      if (scrollDebounceTimer) {
        clearTimeout(scrollDebounceTimer);
      }
      
      scrollDebounceTimer = setTimeout(() => {
        const scrollDelta = Math.abs(window.scrollY - lastScrollTop);
        
        // Si scroll significatif (plus d'un écran), proposer une réanalyse
        if (scrollDelta > window.innerHeight * 0.8) {
          lastScrollTop = window.scrollY;
          
          // Afficher une notification pour réanalyser
          if (lastResults && currentStatus !== 'checking') {
            showScrollNotification();
          }
        }
      }, CONFIG.SCROLL_DEBOUNCE_MS);
    });
  }

  function showScrollNotification() {
    const existing = document.getElementById('factpulse-scroll-notif');
    if (existing) return;
    
    const notif = document.createElement('div');
    notif.id = 'factpulse-scroll-notif';
    notif.className = 'factpulse-scroll-notification';
    notif.innerHTML = `
      <span>Nouveau contenu visible</span>
      <button id="fp-scroll-analyze">Analyser</button>
      <button id="fp-scroll-dismiss">×</button>
    `;
    
    document.body.appendChild(notif);
    
    setTimeout(() => notif.classList.add('fp-notif-visible'), 10);
    
    document.getElementById('fp-scroll-analyze').addEventListener('click', () => {
      notif.remove();
      runAnalysis();
    });
    
    document.getElementById('fp-scroll-dismiss').addEventListener('click', () => {
      notif.classList.remove('fp-notif-visible');
      setTimeout(() => notif.remove(), 300);
    });
    
    // Auto-dismiss après 5 secondes
    setTimeout(() => {
      if (notif.parentNode) {
        notif.classList.remove('fp-notif-visible');
        setTimeout(() => notif.remove(), 300);
      }
    }, 5000);
  }

  // ============================================================================
  // INITIALISATION
  // ============================================================================
  
  function init() {
    createBadge();
    updateBadge('idle');
    setupScrollObserver();
    
    console.log('[FactPulse] Extension v2.0 initialisée - Analyse viewport');
  }
  
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
  
  // Exposer pour le background script
  window.factpulse = {
    analyze: runAnalysis,
    getStatus: () => currentStatus,
    getResults: () => lastResults,
    clear: clearAllHighlights
  };

  // ============================================================================
  // MESSAGE LISTENER
  // ============================================================================
  
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
      return true;
    }
    
    if (message.action === 'getStatus') {
      sendResponse({ 
        status: currentStatus, 
        results: lastResults 
      });
      return false;
    }
    
    if (message.action === 'clear') {
      clearAllHighlights();
      lastResults = null;
      updateBadge('idle');
      sendResponse({ success: true });
      return false;
    }
    
    return false;
  });

})();
