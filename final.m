%% Part-1
%Insert Shape
warning('off');
pic=imread('pts.jpg');
imshow(pic);

box=[500 500 100 100];
pic2=insertShape(pic,'Rectangle',box,'Linewidth',5,'Color','blue');
imshow(pic2);

polypoints=[500 500 750 250 1000 500 1000 1000 500 1000];
pic3=insertShape(pic,'Polygon',polypoints,'Linewidth',5,'color','yellow');
imshow(pic3);

%Insert Marker
pos=[510,510; 550 590; 560 540];
pic3=insertMarker(pic2,pos,'+','color','yellow','size',10);
imshow(pic3);
%%
%reshape
mat=[1 2 3; 4 5 6]
reshape(mat,[3,2])
reshape(mat,1,[])

%bbox 2 points
box=[10,10,90,90];
pts=bbox2points(box);
%% detect min eigen features
cor=detectMinEigenFeatures(rgb2gray(pic),'ROI',[500 500 600 600]);
% imshow(pic);hold on
% plot(cor);
picpoints=insertMarker(pic,cor,'+');
imshow(picpoints);

%% video player
videoReader = vision.VideoFileReader('visionface.avi'); %vid.mp4
videoPlayer = vision.VideoPlayer;

%cont = ~isDone(videoFReader);
while ~isDone(videoReader)
  frame = step(videoReader);
  step(videoPlayer, frame);
  % Continue the loop until the last frame is read.
  % Exit the loop if the video player figure is closed by user.     
  % cont = ~isDone(videoFReader) && isOpen(videoPlayer);
end
release(videoReader);
release(videoPlayer);

%% Video point tracker
videoFileReader = vision.VideoFileReader('visionface.avi');
videoPlayer = vision.VideoPlayer('Position', [100, 100, 680, 520]);

objectFrame = step(videoFileReader);  % read the first video frame

objectRegion = [264, 122, 93, 93];  
objectImage = insertShape(objectFrame, 'Rectangle', objectRegion, ...
                          'Color', 'red');
figure; imshow(objectImage); title('Red box shows object region');

points = detectMinEigenFeatures(rgb2gray(objectFrame), 'ROI', objectRegion);

pointImage = insertMarker(objectFrame, points.Location, '+', ...
                          'Color', 'white');
figure, imshow(pointImage), title('Detected interest points');
tracker = vision.PointTracker('MaxBidirectionalError', 1);

initialize(tracker, points.Location, objectFrame);
while ~isDone(videoFileReader)
  frame = step(videoFileReader);             % Read next image frame
  [points, validity] = step(tracker, frame);  % Track the points
  out = insertMarker(frame, points(validity, :), '+'); % Display points
  step(videoPlayer, out);                    % Show results
end

release(videoPlayer);
release(videoFileReader);

%% Video point Tracker with Box
% Create System objects for reading and displaying video, and for
    % drawing a bounding box of the object.
    videoFileReader = vision.VideoFileReader('visionface.avi');
    videoPlayer = vision.VideoPlayer();
 
    % Read the first video frame which contains the object and then show
    % the object region
    objectFrame = step(videoFileReader);  % read the first video frame
  
    objectRegion = [264, 122, 93, 93];  % define the object region
    % You can also use the following commands to select the object region
    % using a mouse. The object must occupy majority of the region.
    % figure; imshow(objectFrame); objectRegion=round(getPosition(imrect))
    bboxPoints = bbox2points(objectRegion(1, :));
    bboxPolygon = reshape(bboxPoints',1,[]);
    objectImage = insertShape(objectFrame, 'Polygon',bboxPolygon, ...
                              'Color', 'red');
    figure; imshow(objectImage); title('Red box shows object region');
  
    % Detect interest points in the object region
    points = detectMinEigenFeatures(rgb2gray(objectFrame), 'ROI', objectRegion);
    
    xyPoints = points.Location;
    
    oldPoints = xyPoints;
    % Display the detected points
    pointImage = insertMarker(objectFrame, points.Location, '+', ...
                              'Color', 'white');
    figure, imshow(pointImage), title('Detected interest points');
  
    % Create a tracker object.
    pointTracker = vision.PointTracker('MaxBidirectionalError', 1);
  
    % Initialize the tracker
    initialize(pointTracker, points.Location, objectFrame);
  
    % Track and display the points in each video frame
    while ~isDone(videoFileReader)
        frame = step(videoFileReader);             % Read next image frame
        [xyPoints, isFound] = step(pointTracker, frame);
        visiblePoints = xyPoints(isFound, :);
        oldInliers = oldPoints(isFound, :);

        % Estimate the geometric transformation between the old points
        % and the new points.
        [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
            oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);

        % Apply the transformation to the bounding box.
        bboxPoints = transformPointsForward(xform, bboxPoints);

        % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
        % format required by insertShape.
        bboxPolygon = reshape(bboxPoints', 1, []);

        % Display a bounding box around the face being tracked.
        videoFrame = insertShape(frame, 'Polygon', bboxPolygon, 'LineWidth', 3);

        % Display tracked points.
        videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'white');

        % Reset the points.
        oldPoints = visiblePoints;
        setPoints(pointTracker, oldPoints);
        step(videoPlayer, videoFrame);                    % Show results
   end
  
   release(videoPlayer);
   release(videoFileReader);

%% Webcam point tracker
% Create the webcam object.
if exist('cam') ==0
cam = webcam( );
end

% Capture one frame to get its size.
videoFrame = snapshot(cam);
frameSize = size(videoFrame);

% Create the video player object.
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+10]);

objectRegion = [521,108, 400, 400];  
objectImage = insertShape(videoFrame, 'Rectangle', objectRegion, ...
                          'Color', 'red');
figure; imshow(objectImage); title('Red box shows object region');

points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', objectRegion);

pointImage = insertMarker(videoFrame, points.Location, '+', ...
                          'Color', 'white');
figure, imshow(pointImage), title('Detected interest points');
tracker = vision.PointTracker('MaxBidirectionalError', 1);

initialize(tracker, points.Location, videoFrame);
step(videoPlayer, pointImage);
while isOpen(videoPlayer)
  frame = snapshot(cam);             % Read next image frame
  [points, validity] = step(tracker, frame);  % Track the points
  out = insertMarker(frame, points(validity, :), '+'); % Display points
  step(videoPlayer, out);                    % Show results
end

release(videoPlayer);
release(videoFileReader);

%% Webcam point tracker with face detection
% Create the webcam object.
if exist('cam') ==0
cam = webcam( );
end

% Capture one frame to get its size.
videoFrame = snapshot(cam);
frameSize = size(videoFrame);

%face detection to get ROI
faceDetector = vision.CascadeObjectDetector();

% Create the video player object. 
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+10]);

%objectRegion = [521,108, 400, 400];  
objectRegion = faceDetector.step(rgb2gray(videoFrame));

objectImage = insertShape(videoFrame, 'Rectangle', objectRegion, ...
                          'Color', 'red');
figure; imshow(objectImage); title('Red box shows object region');

points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', objectRegion);

pointImage = insertMarker(videoFrame, points.Location, '+', ...
                          'Color', 'white');
figure, imshow(pointImage), title('Detected interest points');
tracker = vision.PointTracker('MaxBidirectionalError', 1);

initialize(tracker, points.Location, videoFrame);
step(videoPlayer, pointImage);
while isOpen(videoPlayer)
  frame = snapshot(cam);             % Read next image frame
  [points, validity] = step(tracker, frame);  % Track the points
  out = insertMarker(frame, points(validity, :), '+'); % Display points
  step(videoPlayer, out);                    % Show results
end

release(videoPlayer);

%% Face detection and point tracking

% Create the face detector object.
faceDetector = vision.CascadeObjectDetector();

% Create the point tracker object.
pointTracker = vision.PointTracker('MaxBidirectionalError', 3);

% Create the webcam object.
if exist('cam') ==0
cam = webcam( );
end

% Capture one frame to get its size.
videoFrame = snapshot(cam);
frameSize = size(videoFrame);

% Create the video player object.
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+10]);

runLoop = true;
numPts = 0;
frameCount = 0;

while runLoop && frameCount < 1000

    % Get the next frame.
    videoFrame = snapshot(cam);
    
    % Get frame to save data to database
    videoFrameGray = rgb2gray(videoFrame);
    frameCount = frameCount + 1;

    if numPts < 10
        % Detection mode.
        bbox = faceDetector.step(videoFrameGray);

        if ~isempty(bbox)
            % Find corner points inside the detected region.
            points = detectMinEigenFeatures(videoFrameGray, 'ROI', bbox(1, :));

            % Re-initialize the point tracker.
            xyPoints = points.Location;
            numPts = size(xyPoints,1);
            release(pointTracker);
            initialize(pointTracker, xyPoints, videoFrameGray);

            % Save a copy of the points.
            oldPoints = xyPoints;

            % Convert the rectangle represented as [x, y, w, h] into an
            % M-by-2 matrix of [x,y] coordinates of the four corners. This
            % is needed to be able to transform the bounding box to display
            % the orientation of the face.
            bboxPoints = bbox2points(bbox(1, :));

            % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
            % format required by insertShape.
            bboxPolygon = reshape(bboxPoints', 1, []);

            % Display a bounding box around the detected face.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);

            % Display detected corners.
            videoFrame = insertMarker(videoFrame, xyPoints, '+', 'Color', 'white');
        end

    else
        % Tracking mode.
        [xyPoints, isFound] = step(pointTracker, videoFrameGray);
        visiblePoints = xyPoints(isFound, :);
        oldInliers = oldPoints(isFound, :);

        numPts = size(visiblePoints, 1);

        if numPts >= 10
            % Estimate the geometric transformation between the old points
            % and the new points.
            [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);

            % Apply the transformation to the bounding box.
            bboxPoints = transformPointsForward(xform, bboxPoints);

            % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
            % format required by insertShape.
            bboxPolygon = reshape(bboxPoints', 1, []);
            
            % Display a bounding box around the face being tracked.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);

            % Display tracked points.
            videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'white');

            % Reset the points.
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
        end

    end

    % Display the annotated video frame using the video player object.
    step(videoPlayer, videoFrame);

    % Check whether the video player window has been closed.
    runLoop = isOpen(videoPlayer);
end

% Clean up.
clear cam;
release(videoPlayer);
release(pointTracker);
release(faceDetector);