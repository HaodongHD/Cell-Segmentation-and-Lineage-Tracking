function results = segment_cells_from_ilastik(h5Path, datasetPath, originalTifPath, opts)
%SEGMENT_CELLS_FROM_ILASTIK  Segment cells & extract centroids from ilastik output.
%
% results = segment_cells_from_ilastik(h5Path, datasetPath, originalTifPath, opts)
%
% Inputs
% -------
% h5Path          : char/string. Path to .h5 exported by ilastik (binary/prob map).
% datasetPath     : char/string. HDF5 dataset path (e.g. '/exported_data').
% originalTifPath : char/string or []. Optional path to original TIFF for plotting overlays.
% opts            : struct of options (all optional; sensible defaults below).
%
%   opts.isProbability   (logical)   : true if H5 is probabilities; false if already binary. [true]
%   opts.probThreshold   (double)    : threshold in [0,1] if isProbability=true. [] => Otsu per-frame. []
%   opts.pix2micron      (double)    : pixel size in micrometers. [1.0]
%   opts.roiCenter_um    (1x2 double): [x_um, y_um] ROI center in micrometers. [[0 0]]
%   opts.roiRadius_um    (double)    : ROI radius in micrometers. [inf] (no ROI)
%   opts.doWatershed     (logical)   : split touching cells with DT+watershed. [true]
%   opts.hMin            (double)    : H-minima depth for watershed (in pixels). [2]
%   opts.minArea_px      (double)    : remove objects smaller than this area (px). [20]
%   opts.maxArea_px      (double)    : remove objects larger than this area (px). [inf]
%   opts.connectivity    (double)    : 4 or 8 connectivity. [8]
%   opts.plotOverlays    (logical)   : save per-frame overlays (requires originalTifPath). [false]
%   opts.overlayDir      (char)      : directory for overlays (created if missing). ['overlays']
%   opts.overlayMarkerSz (double)    : scatter size in pixels. [6]
%   opts.saveMat         (logical)   : save results to MAT file. [true]
%   opts.matOutPath      (char)      : path to MAT file. ['segmentation_results.mat']
%   opts.useParallel     (logical)   : use PARFOR for per-frame loops. [true]
%   opts.plotSummaries   (logical)   : create summary plots of counts. [false]
%
% Outputs
% -------
% results : struct with fields
%   .labels_TYX      : int32 [T x Y x X] label images (0=background).
%   .centroids_px    : 1xT cell, each Nx2 [x_px,y_px] centroids (all cells).
%   .centroids_um    : 1xT cell, each Nx2 [x_um,y_um] centroids (all cells).
%   .centroids_roi_um: 1xT cell, each Mx2 [x_um,y_um] centroids inside ROI.
%   .count_total     : [T x 1] total objects per frame (area-filtered).
%   .count_roi       : [T x 1] objects inside ROI per frame.
%   .count_increase  : [T-1 x 1] delta count_roi(t) = count_roi(t+1)-count_roi(t).
%   .count_increase_density : [T-1 x 1] delta / count_roi(t).
%   .opts            : options struct actually used.
%
% Example
% -------
% opts = struct('pix2micron',0.69,'roiCenter_um',[0 0],'roiRadius_um',5000, ...
%               'plotOverlays',true,'overlayDir','seg_overlays');
% R = segment_cells_from_ilastik('somite_july_short_binary.h5','/exported_data','Somite_July_short_green.tif',opts);

% Timer for performance info
wallStart = tic; %#ok<TNMLP>

% -------------------------- Defaults & Validation --------------------------
if nargin < 4, opts = struct(); end
opts = setDefault(opts, 'isProbability',   true);
opts = setDefault(opts, 'probThreshold',   []);
opts = setDefault(opts, 'pix2micron',      1.0);
opts = setDefault(opts, 'roiCenter_um',    [0 0]);
opts = setDefault(opts, 'roiRadius_um',    inf);
opts = setDefault(opts, 'doWatershed',     true);
opts = setDefault(opts, 'hMin',            2);
opts = setDefault(opts, 'minArea_px',      20);
opts = setDefault(opts, 'maxArea_px',      inf);
opts = setDefault(opts, 'connectivity',    8);
opts = setDefault(opts, 'plotOverlays',    false);
opts = setDefault(opts, 'overlayDir',      'overlays');
opts = setDefault(opts, 'overlayMarkerSz', 6);
opts = setDefault(opts, 'saveMat',         true);
opts = setDefault(opts, 'matOutPath',      'segmentation_results.mat');
opts = setDefault(opts, 'useParallel',     true);
opts = setDefault(opts, 'plotSummaries',   false);

validateattributes(h5Path, {'char','string'}, {'nonempty'});
validateattributes(datasetPath, {'char','string'}, {'nonempty'});
if ~isempty(originalTifPath)
    validateattributes(originalTifPath, {'char','string'}, {'nonempty'});
end
if opts.plotOverlays && isempty(originalTifPath)
    error('plotOverlays=true requires originalTifPath.');
end
if opts.plotOverlays && ~exist(opts.overlayDir,'dir')
    mkdir(opts.overlayDir);
end

% ------------------------------ Load H5 Stack ------------------------------
raw = h5read(char(h5Path), char(datasetPath));  % ilastik often exports as [C,Y,X,T] or [Y,X,T]
sz  = size(raw);
switch numel(sz)
    case 3  % [Y,X,T]
        data = raw;
    case 4  % [C,Y,X,T] -> take first channel/class
        data = squeeze(raw(1,:,:,:));
    otherwise
        error('Unsupported H5 dimensionality: %s', mat2str(sz));
end
[Y, X, T] = size(data);

% ------------------------ Binary Mask (Thresholding) -----------------------
% If probabilities in [0,1], convert to logical masks. If already binary, just cast.
if opts.isProbability
    masks = false(Y, X, T);
    for t = 1:T
        img = data(:,:,t);
        if isempty(opts.probThreshold)
            % Otsu per frame; small bias toward foreground for probability maps
            thr = graythresh(img);
            thr = max(min(thr*0.95, 0.99), 0.01);
        else
            thr = opts.probThreshold;
        end
        masks(:,:,t) = img >= thr;
    end
else
    masks = logical(data);
end

% ----------------- Morphology Cleanup + Area Filtering ---------------------
for t = 1:T
    bw = masks(:,:,t);
    bw = imfill(bw, 'holes');
    % Remove small objects (min area)
    bw = bwareaopen(bw, max(1, round(opts.minArea_px)));
    if isfinite(opts.maxArea_px)
        % Remove objects larger than maxArea: keep only labels within [min,max]
        CC = bwconncomp(bw, opts.connectivity);
        stats = regionprops(CC, 'Area');
        keep = find([stats.Area] >= opts.minArea_px & [stats.Area] <= opts.maxArea_px);
        if ~isempty(keep)
            bw = ismember(labelmatrix(CC), keep);
        else
            bw = false(size(bw));
        end
    end
    masks(:,:,t) = bw;
end

% --------------------------- Watershed Splitting ---------------------------
labels_TYX = zeros(T, Y, X, 'int32');
usePar = opts.useParallel && ~isempty(ver('parallel'));

if usePar
    parfor t = 1:T
        labels_TYX(t,:,:) = segmentFrame(masks(:,:,t), opts);
    end
else
    for t = 1:T
        labels_TYX(t,:,:) = segmentFrame(masks(:,:,t), opts);
    end
end

% ------------------------- Centroids & ROI Filtering -----------------------
centroids_px     = cell(1, T);
centroids_um     = cell(1, T);
centroids_roi_um = cell(1, T);
count_total      = zeros(T,1);
count_roi        = zeros(T,1);

px2um = opts.pix2micron;
roiCx = opts.roiCenter_um(1);
roiCy = opts.roiCenter_um(2);
roiR  = opts.roiRadius_um;

for t = 1:T
    L   = squeeze(labels_TYX(t,:,:));
    if ~any(L(:))
        centroids_px{t}     = zeros(0,2);
        centroids_um{t}     = zeros(0,2);
        centroids_roi_um{t} = zeros(0,2);
        continue;
    end

    stats = regionprops(L, 'Centroid');
    Cpx = vertcat(stats.Centroid);    % [N x 2] in px (order: [x,y])
    Cum = Cpx .* px2um;               % convert to micrometers

    count_total(t)  = size(Cpx,1);

    if isfinite(roiR)
        dx = Cum(:,1) - roiCx;
        dy = Cum(:,2) - roiCy;
        inside = hypot(dx,dy) < roiR;
        CumROI = Cum(inside,:);
    else
        CumROI = Cum;
    end

    count_roi(t)        = size(CumROI,1);
    centroids_px{t}     = Cpx;
    centroids_um{t}     = Cum;
    centroids_roi_um{t} = CumROI;
end

% -------------------- Time-series Increases / Densities --------------------
if T >= 2
    count_increase = count_roi(2:end) - count_roi(1:end-1);
    denom = max(count_roi(1:end-1), 1); % avoid divide-by-zero
    count_increase_density = count_increase ./ denom;
else
    count_increase = [];
    count_increase_density = [];
end

% ------------------------------ Overlays (opt) -----------------------------
if opts.plotOverlays
    for t = 1:T
        I = imread(char(originalTifPath), t);
        f = figure('Visible','off','Color','w');
        imshow(I,[]); hold on;
        Cpx = centroids_px{t}; % plot in pixels on the image grid
        if ~isempty(Cpx)
            scatter(Cpx(:,1), Cpx(:,2), opts.overlayMarkerSz, 'filled', 'MarkerFaceColor',[0 1 0]);
        end
        hold off;
        outName = fullfile(opts.overlayDir, sprintf('cell_center_%04d.tif', t));
        exportgraphics(gca, outName, 'Resolution', 200);
        close(f);
    end
end

% ------------------------------ Summary Plots ------------------------------
if opts.plotSummaries
    f = figure('Color','w','Position',[100 100 1100 400]);
    tiledlayout(1,2,'TileSpacing','compact');

    nexttile; plot(1:T, count_total, '-o'); grid on;
    xlabel('Time point'); ylabel('Number of cells'); title('Cells per frame (total)');

    nexttile; plot(1:T, count_roi, '-o'); grid on;
    xlabel('Time point'); ylabel('Number of cells'); title('Cells per frame (ROI)');
end

% ------------------------------- Save & Return -----------------------------
results = struct();
results.labels_TYX               = labels_TYX;           % [T,Y,X]
results.centroids_px             = centroids_px;         % cell 1xT
results.centroids_um             = centroids_um;         % cell 1xT
results.centroids_roi_um         = centroids_roi_um;     % cell 1xT
results.count_total              = count_total;          % [T,1]
results.count_roi                = count_roi;            % [T,1]
results.count_increase           = count_increase;       % [T-1,1]
results.count_increase_density   = count_increase_density; % [T-1,1]
results.opts                     = opts;                 % options used

if opts.saveMat
    save(opts.matOutPath, 'results', '-v7.3');
end

fprintf('[segment_cells_from_ilastik] Done in %.2f s. Y=%d, X=%d, T=%d\n', toc(wallStart), Y, X, T);
end

% ============================================================================
% Helper: per-frame segmentation using distance transform + watershed
% ----------------------------------------------------------------------------
function L = segmentFrame(bw, opts)
%SEGMENTFRAME  Create label image from a binary mask, optionally splitting with watershed.
% Inputs:
%   bw   : logical 2D mask
%   opts : options struct (needs fields doWatershed, hMin, connectivity)

if ~any(bw(:))
    L = zeros(size(bw), 'int32');
    return;
end

if ~opts.doWatershed
    L = bwlabel(bw, opts.connectivity);
    L = int32(L);
    return;
end

% Distance transform inside objects
D = bwdist(~bw);

% Use negative distance for watershed ridges; suppress shallow minima
Dn = -D;
Dn(~bw) = Inf;                 % ensure background is excluded
Dn = imhmin(Dn, max(0, opts.hMin));
L = watershed(Dn);
L(~bw) = 0;                    % clear background labels

% Re-label to have consecutive integers
if any(L(:))
    L = bwlabel(L > 0, opts.connectivity);
else
    L = zeros(size(bw));
end
L = int32(L);
end

% ============================================================================
% Helper: set default field if not present
% ----------------------------------------------------------------------------
function s = setDefault(s, name, value)
if ~isfield(s, name) || isempty(s.(name))
    s.(name) = value;
end
end
