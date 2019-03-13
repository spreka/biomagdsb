%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GENERATE_OBJECTS Function for generating required amount of objects
% Input:  (1) struct for cell level parameters
%         (2) struct for population level parameters
% Output: (1) struct containing arrays for all different object types
%
% (C) 22.2.2007 Antti Lehmussola
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function[out] = generate_dsb_objects(cell_obj,population)

simcep_dsb_options;

nuclei_vector =[];
cytoplasm_vector =[];
subcell_vector = [];

% Limit for unsuccesfully trying to generate new objects
FAIL_LIMIT = 500;

locations = cell(size(population.template));

%Generate cluster center coordinates

cluster = [randi(size(population.template,1), 1, population.clust)+1;...
    randi(size(population.template,2), 1, population.clust)+1];

%Generate requited amount objects
for ind = 1:population.N
    
    if mod(ind, population.N/10)==0
        fprintf('#');
    end
    
    %Status for succesful object generation
    status = 0;
    
    %Amount of unsuccesfully generated objects
    failed = 0;
    
    %Stay here until object is generated succesfully
    while status == 0
        
        % CSABA'S MODIFICATIONS TO CONTROL POSITIONS
        
        %Assign coordinates for each object
        
%         ls = cellfun(@(x) isempty(x), locations);
%         freeIdxList = find(ls>0);
        
        if rand < population.clustprob
            %Select randomly some cluster center
            C = randi(population.clust,1,1);
            Y = round(cluster(1,C)+randn*population.spatvar*size(population.template,1));
            X = round(cluster(2,C)+randn*population.spatvar*size(population.template,2));
        else
            Y = randi(size(population.template,1), 1, 1);
            X = randi(size(population.template,2), 1, 1);
%             newPos = freeIdxList(randi(length(freeIdxList)));
%             [X,Y] = ind2sub(size(population.template), newPos);
        end
        
        if population.overlap_obj == 1 %Overlap controlled in nucleus level
            if cell_obj.nucleus.include == 0
                error('Impossible to control nuclei overlap since nuclei simulation is not selected');
            end
            
            selectedCellId = randi(size(cell_obj.nucleus.representativeFeatureVectors,1));
            
            %Generate new nucleus
            n = nucleus([Y X], ind, sqrt(cell_obj.nucleus.representativeFeatureVectors(selectedCellId,1)/pi),cell_obj.nucleus.shape,...
                cell_obj.nucleus.texture);
            
            shape = getshape(n);
            
            shapeSize = size(shape);
            
            shapeRatio = cell_obj.nucleus.representativeFeatureVectors(selectedCellId,2);
            
            shape = imresize(shape, [shapeSize(1)/sqrt(shapeRatio), shapeSize(2)*sqrt(shapeRatio)]);
            
            rotAngle = randi(360);
            shape = imrotate(im2double(shape), rotAngle, 'bilinear')>0.5;
            
            emptyRows = ~any(shape,2);
            shape(emptyRows,:) = [];

            emptyCols = ~any(shape,1);
            shape(:,emptyCols) = [];
            
            object = shape;
            
            n2 = cellobj(getcoords(n), ind, shape, object, struct('coords',getcoords(n),'area',sum(shape(:))));
            
            [locations,status] = add_object_dsb(n2,locations,population.overlap,cell_obj.nucleus.representativeFeatureVectors(selectedCellId,:));
            
            if status == 0
                failed = failed + 1;
                continue;
            end
            nuclei_vector = [nuclei_vector n2];
            
        elseif population.overlap_obj == 2 %Overlap controlled in cytoplasm level
            if cell_obj.cytoplasm.include == 0
                error('Impossible to control cytoplasm overlap since cytoplasm simulation is not selected');
            end
            
            %Generate new cytoplasm
            c = cytoplasm([Y X], ind, cell_obj.cytoplasm.radius,cell_obj.cytoplasm.shape,...
                cell_obj.cytoplasm.texture);
            [locations,status] = add_object(c,locations,population.overlap);
            if status == 0
                failed = failed + 1;
                continue;
            end
            cytoplasm_vector = [cytoplasm_vector c];
        end
        
        
        if failed > FAIL_LIMIT %Image is too full to add new objects
            warning(['Not enough space for new objects. Only ' num2str(eval('ind')-1) ...
                ' objects were generated. Wait until simulation is finished...']);
            break;
        end
        
%         failed = failed + 1;
        
    end
    
    %Limit for how many times we try to generate one cell
    if failed > FAIL_LIMIT
        break;
    end
    
    %Generate other required objects
    if cell_obj.cytoplasm.include == 1 & population.overlap_obj == 1
        c = cytoplasm([Y X], ind, cell_obj.cytoplasm.radius,cell_obj.cytoplasm.shape,...
            cell_obj.cytoplasm.texture);
        cytoplasm_vector = [cytoplasm_vector c];
    end
    if cell_obj.nucleus.include == 1 & population.overlap_obj == 2
        n = nucleus([Y X], ind, cell_obj.nucleus.radius,cell_obj.nucleus.shape,...
            cell_obj.nucleus.texture);
        nuclei_vector = [nuclei_vector n];
    end
    
    if cell_obj.subcell.include == 1 & cell_obj.cytoplasm.include == 1
        s = subcell([Y X],ind,cell_obj.cytoplasm.radius/10,cell_obj.subcell.ns,c);
        subcell_vector = [subcell_vector s];
    end
    
end

out.nuclei = nuclei_vector;
out.cytoplasm = cytoplasm_vector;
out.subcell = subcell_vector;
