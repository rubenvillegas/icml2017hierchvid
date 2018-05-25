for i=1:11                                                                       
  data_path = ['./S' num2str(i) '/MyPoseFeatures/D2_Positions/'];
  disp(data_path);                                                               
  pose_list = dir([data_path '*cdf']);                                           
  for j=1:length(pose_list)                                                      
    data = cdfread([data_path pose_list(j).name]);                               
    data = data{1};                                                              
    save([data_path pose_list(j).name(1:end-4) '.mat'], 'data');                 
  end                                                                            
end
