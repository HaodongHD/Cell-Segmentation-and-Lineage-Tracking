function R = track_cell_lineages(r_c, originalTifPath, opts)
%TRACK_CELL_LINEAGES  Frame-to-frame cell lineage tracking from centroids.
%
%   R = track_cell_lineages(r_c, originalTifPath, opts)
%
% Inputs
% ------
% r_c              : 1xT cell array. r_c{t} is [N_t x 2] centroid positions.
%                    Positions can be in micrometers or pixels (see opts.positionsAreMicrons).
% originalTifPath  : char/string or []. Optional path to original TIFF stack for overlays.
% opts             : (struct) optional parameters with fields:
%   positionsAreMicrons (logical) : true if r_c are in micrometers. [true]
%   pix2micron          (double)  : pixel size (µm/px), used for overlays. [1.0]
%   matchingMode        (char)    : 'one2one' | 'many2one'. ["many2one"]
%                                   'one2one' uses optimal linear assignment (matchpairs)
%                                   'many2one' assigns each t+1 cell to its nearest t parent (allows merges).
%   maxLinkDistance_um  (double)  : maximum allowed link distance in micrometers. [Inf]
%   useParallel         (logical) : use PARFOR for overlays. [true]
%   plotOverlays        (logical) : save per-frame colored overlays. Requires originalTifPath. [false]
%   overlayDir          (char)    : directory to save overlays. ['tracking_overlays']
%   overlayMarkerSz     (double)  : scatter marker size (px). [6]
%   saveMat             (logical) : save struct R to MAT file. [true]
%   matOutPath          (char)    : path to save MAT. ['tracking_results.mat']
%   seedColors          (double)  : Kx3 array of RGB colors to reuse; otherwise auto-generated. []
%
% Outputs
% -------
% R : struct with fields
%   .cell_lineage_index : cell [L x T]. row i = lineage i; entry is vector
%                         of indices into r_c{t} for that lineage at time t.
%                         L = number of cells at t=1.
%   .index_adjacent_time: 1x(T-1) cell; mapping from t+1 -> t (length N_{t+1}).
%                         For many2one: map(j) = parent index at t; 0 if unlinked (too far).
%                         For one2one : same, but unique parent/child pairs.
%   .colors_rgb         : [L x 3] RGB colors for each lineage.
%   .opts               : (struct) options actually used.
%
% Notes
% -----
% - This tracker assumes *no* appearance before t=1 for lineages and defines
%   the number of lineages as N_1 = size(r_c{1},1). New cells appearing later
%   are ignored for new-lineage creation (but are mapped to nearest parent in
%   many2one mode if within maxLinkDistance_um).
% - If your r_c are in pixels, set opts.positionsAreMicrons=false and ensure
%   pix2micron is provided for proper overlay scaling/limits.
%
% Example
% -------
% opts = struct('positionsAreMicrons',true,'pix2micron',0.69,'plotOverlays',true);
% R = track_cell_lineages(r_c, 'Somite_July_short_green.tif', opts);

% -------------------------------------------------------------------------
% Setup & validation
% -------------------------------------------------------------------------
wallStart = tic; %#ok<TNMLP>
if nargin < 3, opts = struct(); end
opts = setDefault(opts, 'positionsAreMicrons', true);
opts = setDefault(opts, 'pix2micron',        1.0);
opts = setDefault(opts, 'matchingMode',      'many2one');
opts = setDefault(opts, 'maxLinkDistance_um', inf);
opts = setDefault(opts, 'useParallel',       true);
opts = setDefault(opts, 'plotOverlays',      false);
opts = setDefault(opts, 'overlayDir',        'tracking_overlays');
opts = setDefault(opts, 'overlayMarkerSz',   6);
opts = setDefault(opts, 'saveMat',           true);
opts = setDefault(opts, 'matOutPath',        'tracking_results.mat');
opts = setDefault(opts, 'seedColors',        []);

validateattributes(r_c, {'cell'}, {'row'});
T = numel(r_c);
assert(T >= 1, 'r_c must contain at least one frame');
for t = 1:T
    if ~isempty(r_c{t})
        validateattributes(r_c{t}, {'double','single'}, {'ncols',2});
    end
end
if opts.plotOverlays
    if isempty(originalTifPath)
        error('plotOverlays=true requires originalTifPath.');
    end
    if ~exist(opts.overlayDir,'dir'), mkdir(opts.overlayDir); end
end

% -------------------------------------------------------------------------
% Initialize lineages
% -------------------------------------------------------------------------
num_time_points   = T;
num_cell_lineage  = sizeSafe(r_c{1},1);              % L := cells at t=1
cell_lineage_index = cell(num_cell_lineage, T);      % L x T
index_adjacent_time = cell(1, T-1);                  % map t+1 -> t

% Initial lineage IDs = 1..L linked to their index at t=1
for i = 1:num_cell_lineage
    if i <= sizeSafe(r_c{1},1)
        cell_lineage_index{i,1} = i;
    else
        cell_lineage_index{i,1} = [];
    end
end
% Initialize mapping for t=1 as identity for existing cells
index_adjacent_time{1} = (1:sizeSafe(r_c{1},1)).';

% -------------------------------------------------------------------------
% Build frame-to-frame correspondences
% -------------------------------------------------------------------------
for t = 1:(num_time_points-1)
    X = r_c{t};      % [N1 x 2]
    Y = r_c{t+1};    % [N2 x 2]
    N1 = sizeSafe(X,1);
    N2 = sizeSafe(Y,1);

    if N1==0 || N2==0
        index_adjacent_time{t+1} = zeros(N2,1);
        % propagate empties
        for i = 1:num_cell_lineage
            cell_lineage_index{i,t+1} = [];
        end
        continue;
    end

    % Convert to micrometers if needed for distance thresholding
    if ~opts.positionsAreMicrons
        px2um = opts.pix2micron;
        Xum = X .* px2um;
        Yum = Y .* px2um;
    else
        Xum = X; Yum = Y;
    end

    % Compute pairwise squared distances (um^2)
    % Efficiently: ||x-y||^2 = |x|^2 + |y|^2 - 2 x·y
    X2 = sum(Xum.^2,2);
    Y2 = sum(Yum.^2,2).';
    D2 = X2 + Y2 - 2*(Xum*Yum.');
    D2 = max(D2,0); % numerical safety
    D  = sqrt(D2);

    % Apply matching mode
    switch lower(opts.matchingMode)
        case 'one2one'
            % Use matchpairs if available; otherwise fall back to greedy
            maxCost = opts.maxLinkDistance_um;
            if isfinite(maxCost)
                C = D; C(C>maxCost) = Inf; % disallow links beyond threshold
            else
                C = D;
            end
            [pairs, ~] = tryMatchPairs(C);
            % pairs: [i_in_X, j_in_Y]
            corr_arr = zeros(N2,1); % map Y(j) -> X(i)
            for k = 1:size(pairs,1)
                i = pairs(k,1); j = pairs(k,2);
                corr_arr(j) = i;
            end
        case 'many2one'
            % Each Y(j) chooses nearest X(i) if within threshold
            [mind, idx] = min(D, [], 1); % over rows (X) for each column j in Y
            corr_arr = idx(:);
            if isfinite(opts.maxLinkDistance_um)
                corr_arr(mind(:) > opts.maxLinkDistance_um) = 0; % 0 => unlinked
            end
        otherwise
            error('Unknown matchingMode: %s', opts.matchingMode);
    end

    % Store mapping t+1 -> t
    index_adjacent_time{t+1} = corr_arr;

    % Update each lineage at t+1 by pulling children of members at t
    for i = 1:num_cell_lineage
        parentIdxs = cell_lineage_index{i,t}; % indices in X (time t)
        if isempty(parentIdxs)
            cell_lineage_index{i,t+1} = [];
            continue;
        end
        % All children in Y that map to any parent in this lineage
        isChild = ismember(corr_arr, parentIdxs);
        kids = find(isChild); % indices in Y (time t+1)
        cell_lineage_index{i,t+1} = kids(:).';
    end
end

% -------------------------------------------------------------------------
% Colors per lineage
% -------------------------------------------------------------------------
if ~isempty(opts.seedColors)
    colors_rgb = repmat(opts.seedColors, ceil(num_cell_lineage/size(opts.seedColors,1)), 1);
    colors_rgb = colors_rgb(1:num_cell_lineage, :);
else
    colors_rgb = makeColors(num_cell_lineage);
end

% -------------------------------------------------------------------------
% Optional overlays
% -------------------------------------------------------------------------
if opts.plotOverlays
    usePar = opts.useParallel && ~isempty(ver('parallel'));
    if usePar
        parfor t = 1:(num_time_points-1)
            saveOverlay(originalTifPath, t, r_c, cell_lineage_index, colors_rgb, opts);
        end
    else
        for t = 1:(num_time_points-1)
            saveOverlay(originalTifPath, t, r_c, cell_lineage_index, colors_rgb, opts);
        end
    end
end

% -------------------------------------------------------------------------
% Pack results
% -------------------------------------------------------------------------
R = struct();
R.cell_lineage_index  = cell_lineage_index;   % [L x T] cell
R.index_adjacent_time = index_adjacent_time;  % 1x(T-1) cell
R.colors_rgb          = colors_rgb;           % [L x 3]
R.opts                = opts;

if opts.saveMat
    save(opts.matOutPath, 'R', '-v7.3');
end

fprintf('[track_cell_lineages] Done in %.2f s. T=%d, L=%d\n', toc(wallStart), T, num_cell_lineage);
end

% ============================================================================
% Helper: attempt matchpairs (SMT) else greedy fallback
% ----------------------------------------------------------------------------
function [pairs, cost] = tryMatchPairs(C)
% TRYMATCHPAIRS  Wrapper that uses matchpairs if available; else greedy.
% C is a cost matrix (rows=X, cols=Y) with Inf for disallowed edges.
try
    [pairs, cost] = matchpairs(C, Inf); % Statistics & ML Toolbox
catch
    % Greedy fallback: repeatedly choose smallest cost remaining
    pairs = zeros(0,2); cost = 0;
    Cwork = C;
    while true
        [m, idx] = min(Cwork(:));
        if isempty(m) || ~isfinite(m), break; end
        [i, j] = ind2sub(size(Cwork), idx);
        pairs(end+1,:) = [i, j]; %#ok<AGROW>
        cost = cost + m;
        Cwork(i,:) = Inf; Cwork(:,j) = Inf; % enforce one-to-one
    end
end
end

% ============================================================================
% Helper: overlay saver for one frame
% ----------------------------------------------------------------------------
function saveOverlay(originalTifPath, t, r_c, cell_lineage_index, colors_rgb, opts)
I = imread(char(originalTifPath), t);
f = figure('Visible','off','Color','w');
imshow(I,[]); hold on;
L = size(cell_lineage_index,1);
for i = 1:L
    idxs = cell_lineage_index{i,t};
    if isempty(idxs), continue; end
    xy  = r_c{t}(idxs, :);
    if opts.positionsAreMicrons
        xy_px = xy ./ opts.pix2micron; % convert µm -> px for display
    else
        xy_px = xy;                    % already in px
    end
    scatter(xy_px(:,1), xy_px(:,2), opts.overlayMarkerSz, 'filled', 'MarkerFaceColor', colors_rgb(i,:));
end
hold off;
outName = fullfile(opts.overlayDir, sprintf('cell_tracking_%04d.tif', t));
exportgraphics(gca, outName, 'Resolution', 200);
close(f);
end

% ============================================================================
% Helper: robust color map (fallback if distinguishable_colors not available)
% ----------------------------------------------------------------------------
function colors = makeColors(N)
try
    colors = distinguishable_colors(N, [0 0 0; 1 1 1]);
catch
    % Fallback: HSV spaced, shuffled for contrast
    colors = hsv(N);
    colors = colors(randperm(N), :);
end
end

% ============================================================================
% Helper: set default field if missing/empty
% ----------------------------------------------------------------------------
function s = setDefault(s, name, value)
if ~isfield(s, name) || isempty(s.(name))
    s.(name) = value;
end
end

% ============================================================================
% Helper: safe size first dim
% ----------------------------------------------------------------------------
function n = sizeSafe(A, dim)
if isempty(A), n = 0; else, n = size(A, dim); end
end
